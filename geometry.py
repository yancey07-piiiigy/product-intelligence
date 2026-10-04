"""SIFT + affine RANSAC verification of CLIP's top 100 candidates.
This improves same-view/cropped photo retrieval, not unseen-view SKU recognition.
"""
import json,time
import cv2
import numpy as np
from PIL import Image
from core import ROOT
from vision import ART,GALLERY,canonical,read_image

VERSION='clip100-sift180-ratio075-affine4px-v1'
cv2.setNumThreads(2)


def features(image):
    a=np.asarray(canonical(image))
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    sift=cv2.SIFT_create(nfeatures=180,contrastThreshold=.018)
    k,d=sift.detectAndCompute(gray,None)
    if d is None:return np.empty((0,2),np.float32),np.empty((0,128),np.float32)
    return np.array([p.pt for p in k],np.float32),d.astype('float32')


def support(qp,qd,cp,cd):
    if len(qd)<4 or len(cd)<4:return 0.,0
    matches=cv2.BFMatcher(cv2.NORM_L2).knnMatch(qd,cd,k=2)
    good=[a for a,b in matches if a.distance<.75*b.distance]
    # A many-to-one match can overstate evidence: keep best match per train feature.
    unique={}
    for m in sorted(good,key=lambda m:m.distance):unique.setdefault(m.trainIdx,m)
    good=list(unique.values())
    if len(good)<4:return 0.,0
    src=np.float32([qp[m.queryIdx] for m in good]);dst=np.float32([cp[m.trainIdx] for m in good])
    cv2.setRNGSeed(6201)
    affine,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=1000,confidence=.99)
    if affine is None or mask is None:return 0.,0
    scale=float(np.linalg.norm(affine[:,0]))
    if not .35<scale<2.8:return 0.,0
    n=int(mask.sum())
    if n<4:return 0.,n
    fraction=n/len(good)
    # Fixed formula chosen before final benchmark. Counts are evidence, not probabilities.
    return min(n/20,1)*fraction,n


def build():
    start=time.perf_counter();meta=json.loads((ART/'index.json').read_text());pts=[];desc=[];offset=[0]
    for i,r in enumerate(meta['rows']):
        p,d=features(read_image(GALLERY/f"{r['id']}.jpg"));pts.append(p);desc.append(d.astype('uint8'));offset.append(offset[-1]+len(d))
        if i%1000==0:print(f'Geometric features {i}/{len(meta["rows"])}',flush=True)
    np.savez_compressed(ART/'geometry.npz',points=np.vstack(pts),descriptors=np.vstack(desc),offsets=np.array(offset,dtype='int64'))
    (ART/'geometry.json').write_text(json.dumps({'index_signature':meta['signature'],'version':VERSION,'seconds':time.perf_counter()-start}))


class Verifier:
    def __init__(self,index_signature,rows):
        metadata=json.loads((ART/'geometry.json').read_text())
        if metadata['version']!=VERSION or metadata['index_signature']!=index_signature:raise ValueError('Stale geometric index. Run python geometry.py')
        with np.load(ART/'geometry.npz',allow_pickle=False) as a:
            self.points=a['points'].copy();self.descriptors=a['descriptors'].copy();self.offsets=a['offsets'].copy()
        self.ids={r['id']:i for i,r in enumerate(rows)}
    def rerank(self,image,hits,k=5):
        qp,qd=features(image);rank=[]
        for h in hits:
            i=self.ids[h['record']['id']];start,end=self.offsets[i:i+2]
            strength,n=support(qp,qd,self.points[start:end],self.descriptors[start:end].astype('float32'))
            rank.append({**h,'clip_score':h['score'],'score':round(.7*strength+.3*h['score'],6),'geometric_support':round(strength,6),'geometric_inliers':n})
        return sorted(rank,key=lambda h:(-h['score'],h['record']['id']))[:k]
if __name__=='__main__':build()
