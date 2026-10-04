"""Local CLIP image-to-image retrieval with geometric reranking."""
import hashlib, io, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from core import ROOT, FIELDS, load_catalog

ART = ROOT / 'artifacts'
MODEL = ROOT / 'models/clip-vit-base-patch32'
MODEL_NAME = 'openai/clip-vit-base-patch32'
PREPROCESS = 'rgb-exif-foreground-crop224-v2'
GALLERY = ROOT / 'data/gallery-images'
Image.MAX_IMAGE_PIXELS = 20_000_000


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def read_image(source):
    try:
        image = Image.open(io.BytesIO(source) if isinstance(source,bytes) else source)
        if image.width*image.height > Image.MAX_IMAGE_PIXELS: raise ValueError('Image exceeds 20 megapixels')
        if image.format not in ('JPEG','PNG','WEBP'): raise ValueError('Use JPEG, PNG or WebP')
        image.load()
        return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError('Invalid or oversized image') from e


def canonical(image):
    """Trim pale studio/background padding; fixed rule, never uses labels."""
    a=np.asarray(image)
    foreground=(a.min(axis=2)<205) | ((a.max(axis=2).astype('int16')-a.min(axis=2))>35)
    ys,xs=np.where(foreground)
    if len(xs)>12:
        box=(max(0,int(xs.min())-2),max(0,int(ys.min())-2),min(image.width,int(xs.max())+3),min(image.height,int(ys.max())+3))
        image=image.crop(box)
    return ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color='white')


class Encoder:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch=torch
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        self.device=os.getenv('CLIP_DEVICE') or ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        if not (MODEL/'config.json').exists(): raise ValueError('CLIP weights missing. Run python prepare_project.py download')
        self.model=CLIPModel.from_pretrained(str(MODEL),local_files_only=True).eval().to(self.device)
        self.processor=CLIPProcessor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
        self.prompts=['a photo of a shoe','a photo of sneakers','a photo of sandals','a photo of boots','a photo of high heel shoes',
                      'a photo of a shirt','a photo of trousers','a photo of a bag','a photo of a wrist watch','a photo of sunglasses','a photo of cosmetics','a photo of a person','a photo of a landscape','a photo of a computer','a photo of a bottle']
        with torch.inference_mode():
            args=self.processor(text=self.prompts,return_tensors='pt',padding=True).to(self.device)
            v=self.model.get_text_features(**args)
            self.text_vectors=(v/v.norm(dim=-1,keepdim=True)).cpu().numpy().astype('float32')
    def encode(self,images):
        images=[canonical(i) for i in images]
        with self.torch.inference_mode():
            args=self.processor(images=images,return_tensors='pt',do_center_crop=False).to(self.device)
            v=self.model.get_image_features(**args)
            return (v/v.norm(dim=-1,keepdim=True)).cpu().numpy().astype('float32')
    def domain(self,v):
        scores=self.text_vectors@v
        footwear=float(max(scores[:5])); other=float(max(scores[5:]))
        return {'is_footwear':footwear>other,'footwear_text_similarity':round(footwear,6),'other_text_similarity':round(other,6),'domain_margin':round(footwear-other,6)}


def index_rows():
    rows,bad=load_catalog(ROOT/'data/styles.csv'); by_id={r['id']:r for r in rows}
    protocol=json.loads((ART/'protocol.json').read_text())
    excluded=set(protocol['excluded_image_hashes'])
    subset_path=ROOT/'data/gallery-subset.json'
    if not subset_path.exists():raise ValueError('Gallery subset missing. Run python gallery_subset.py first')
    subset=json.loads(subset_path.read_text());result=[];hashes=[]
    for pid,expected_sha in zip(subset['selected_ids'],subset['image_hashes']):
        row=by_id.get(pid);p=GALLERY/f'{pid}.jpg'
        if row is None or not p.exists():raise ValueError(f'Gallery subset file missing for {pid}')
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha!=expected_sha:raise ValueError(f'Gallery image changed for {pid}')
        if sha in excluded:raise ValueError(f'Unknown evaluation image leaked into gallery: {pid}')
        result.append(row); hashes.append(sha)
    if len(result)!=subset['target']:raise ValueError('Gallery subset size does not match its manifest')
    return result, hashes


def build_index(batch=48):
    ART.mkdir(exist_ok=True)
    started=time.perf_counter(); encoder=Encoder(); rows,hashes=index_rows(); all_vectors=[]
    for start in range(0,len(rows),batch):
        imgs=[read_image(GALLERY/f"{r['id']}.jpg") for r in rows[start:start+batch]]
        all_vectors.append(encoder.encode(imgs))
        if start%(batch*10)==0: print(f'Indexed {min(start+batch,len(rows))}/{len(rows)}',flush=True)
    model_sha=hashlib.sha256((MODEL/'pytorch_model.bin').read_bytes()).hexdigest()
    metadata={'model':MODEL_NAME,'model_sha256':model_sha,'preprocess':PREPROCESS,'rows':rows,'image_hashes':hashes,
              'protocol_hash':digest(json.loads((ART/'protocol.json').read_text())),'device':encoder.device}
    metadata['signature']=digest({k:metadata[k] for k in ('model','model_sha256','preprocess','rows','image_hashes','protocol_hash')})
    metadata['build_seconds']=round(time.perf_counter()-started,3)
    np.savez_compressed(ART/'image_index.npz',clip=np.vstack(all_vectors))
    (ART/'index.json').write_text(json.dumps(metadata,ensure_ascii=False))
    print(f'Index complete: {len(rows)} images, {metadata["build_seconds"]}s',flush=True)


def choose(hits,domain,calibration,signature):
    if not domain['is_footwear']: return False,'out_of_scope','This image does not appear to show footwear. Please upload a shoe photo.'
    if not calibration or calibration.get('signature')!=signature: return False,'uncalibrated','This index has no valid calibration. Similar candidates are shown for inspection only.'
    if hits[0]['score']<calibration['threshold'] or hits[0]['score']-hits[1]['score']<calibration['margin']:
        return False,'uncertain','Candidates are too similar, or the catalog evidence is insufficient. Compare the images before inspecting a candidate record.'
    return True,'accepted','This candidate passed the calibrated threshold. Compare it with the source image; product identity is not guaranteed.'


class VisualSearch:
    def __init__(self,load_model=True):
        self.meta=json.loads((ART/'index.json').read_text()); self.rows=self.meta['rows']; self.signature=self.meta['signature']
        if self.meta['preprocess']!=PREPROCESS: raise ValueError('Stale preprocessing. Rebuild image index.')
        current=load_catalog(ROOT/'data/styles.csv')[0]; current_map={r['id']:r for r in current}
        if any(current_map.get(r['id'])!=r for r in self.rows): raise ValueError('Catalog changed. Rebuild image index.')
        if hashlib.sha256((MODEL/'pytorch_model.bin').read_bytes()).hexdigest()!=self.meta['model_sha256']:
            raise ValueError('Model weights changed. Rebuild image index.')
        if any(hashlib.sha256((GALLERY/f"{r['id']}.jpg").read_bytes()).hexdigest()!=sha for r,sha in zip(self.rows,self.meta['image_hashes'])):
            raise ValueError('Catalog image changed. Rebuild image index.')
        with np.load(ART/'image_index.npz',allow_pickle=False) as f:
            self.vectors=f['clip'].copy()
        if len(self.vectors)!=len(self.rows): raise ValueError('Corrupt image index')
        self.encoder=Encoder() if load_model else None
        from geometry import Verifier, VERSION
        self.verifier=Verifier(self.meta['signature'],self.rows)
        self.signature=digest({'gallery':self.meta['signature'],'verifier':VERSION})
        self.calibration=json.loads((ART/'calibration.json').read_text()) if (ART/'calibration.json').exists() else None
    def rank(self,vector,k=5):
        scores=self.vectors@vector
        order=np.argsort(-scores,kind='stable')[:k]
        return [{'record':self.rows[int(i)],'score':round(float(scores[i]),6)} for i in order]
    def search(self,raw):
        started=time.perf_counter(); image=read_image(raw); vector=self.encoder.encode([image])[0]
        hits=self.verifier.rerank(image,self.rank(vector,k=100)); domain=self.encoder.domain(vector)
        accepted,status,reason=choose(hits,domain,self.calibration,self.signature)
        return {'accepted':accepted,'status':status,'reason':reason,'candidates':hits,'observation':domain,
                'score':hits[0]['score'],'margin':round(hits[0]['score']-hits[1]['score'],6),
                'calibration':self.calibration,'local_seconds':round(time.perf_counter()-started,4),'api_usd':0,
                'model':MODEL_NAME,'device':self.encoder.device,'signature':self.signature}
