"""Reproducible data split, fixed perturbations and local visual index."""
import argparse,csv,hashlib,json,random,shutil
from pathlib import Path
from PIL import Image,ImageEnhance,ImageOps
from core import ROOT,FIELDS,load_catalog
from vision import ART,MODEL,index_rows,build_index,read_image,digest


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))


def perturb(image,index):
    # Fixed before inference: varied rotation, crop, colour and surrounding canvas.
    r=random.Random(6201+index)
    w,h=image.size; crop=r.uniform(.015,.07)
    im=image.crop((int(w*crop),int(h*crop),w-int(w*crop),h-int(h*crop)))
    im=ImageEnhance.Brightness(im).enhance(r.uniform(.75,1.18))
    im=ImageEnhance.Color(im).enhance(r.uniform(.75,1.15))
    im=im.rotate(r.choice([-9,-5,5,9]),resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(239,236,229))
    return ImageOps.expand(im,border=r.randint(4,12),fill=(239,236,229))


def prepare():
    ART.mkdir(exist_ok=True); rows,_=load_catalog(ROOT/'data/styles.csv'); rng=random.Random(6203)
    development=json.loads((ART/'development-exclusions.json').read_text()) if (ART/'development-exclusions.json').exists() else []
    rng.shuffle(rows); used=set(development); unique=[]
    for row in rows:
        path=ROOT/'data/images'/f"{row['id']}.jpg"
        if not path.exists(): continue
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in used: unique.append((row,sha));used.add(sha)
    known=unique[:70];unknown=unique[70:85]
    other=[]
    with (ROOT/'data/styles.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row.get('masterCategory') not in ('Apparel','Accessories','Personal Care'):continue
            path=ROOT/'data/images'/f"{row['id']}.jpg"
            if path.exists(): other.append(row)
    rng.shuffle(other); outsiders=[]
    for row in other:
        sha=hashlib.sha256((ROOT/'data/images'/f"{row['id']}.jpg").read_bytes()).hexdigest()
        if sha not in used: outsiders.append((row,sha));used.add(sha)
        if len(outsiders)==15:break
    manifest=[]
    for kind,items,cal_n in [('known',known,28),('unknown_footwear',unknown,6),('non_footwear',outsiders,6)]:
        for i,(row,sha) in enumerate(items):
            manifest.append({'query_id':f'q{len(manifest)+1:03}','source_id':row['id'],'expected_id':row['id'] if kind=='known' else None,'kind':kind,'split':'calibration' if i<cal_n else 'test',
                             'original':f'data/images/{row["id"]}.jpg','image':f'artifacts/queries/q{len(manifest)+1:03}.jpg','sha256':sha,
                             'ground_truth':{k:row.get(k,'') for k in FIELDS}})
    protocol={'seed':6203,'version':'stress-v3-frozen','development_exclusions':development,'n':100,'calibration':40,'test':60,'known':70,'unknown_footwear':15,'non_footwear':15,
              'excluded_image_hashes':[s for _,s in unknown], 'description':'Fixed transformed catalog queries for a closed-gallery robustness experiment.'}
    save(ART/'protocol.json',protocol);save(ART/'manifest.json',manifest)
    (ART/'queries').mkdir(exist_ok=True)
    for i,row in enumerate(manifest): perturb(read_image(ROOT/row['original']),i).save(ROOT/row['image'],quality=78)
    print('Prepared 100 fixed queries (70 known / 15 unknown footwear / 15 non-footwear), 40/60 split. Labels saved before inference.')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','prepare','subset','index']);p.add_argument('--batch',type=int,default=48);p.add_argument('--target',type=int,default=2000);a=p.parse_args()
    if a.command=='download':
        from huggingface_hub import snapshot_download
        snapshot_download('openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',local_dir=MODEL,allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','tokenizer.json'])
    elif a.command=='prepare':prepare()
    elif a.command=='subset':
        from gallery_subset import build_subset
        build_subset(target=a.target)
    else:build_index(a.batch)
if __name__=='__main__':main()
