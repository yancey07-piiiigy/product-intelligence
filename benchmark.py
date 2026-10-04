"""Measured, repeatable robustness benchmark. Never substitutes for human study."""
import argparse,json,math,statistics,time
from pathlib import Path
import numpy as np
from core import ROOT,FIELDS
from calibration import select_threshold
from hash_baseline import HandcraftedBaseline
from prepare_project import save
from vision import ART,GALLERY,VisualSearch,read_image,digest
from user_study import study as analyze_user_study


def proportion(n,d):return n/d if d else None

def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]

def metrics(rows,cal):
    known=[r for r in rows if r['kind']=='known']
    accepted=[r for r in rows if r['is_footwear'] and r['score']>=cal['threshold'] and r['margin']>=cal['margin']]
    rejected=[r for r in rows if r not in accepted]; wrong=[r for r in rows if not r['correct']]
    right=sum(r['correct'] for r in accepted);fields=[]
    for r in accepted:
        fields.extend(r['predicted_facts'].get(k)==r['ground_truth'][k] for k in FIELDS if r['ground_truth'].get(k))
    return {'n':len(rows),'known_n':len(known),'accepted_n':len(accepted),'correct_accepted_n':right,
        'known_top1':proportion(sum(r['correct'] for r in known),len(known)),
        'known_top3':proportion(sum(r['expected_id'] in r['top3'] for r in known),len(known)),
        'coverage':proportion(len(accepted),len(rows)),'abstention_rate':proportion(len(rejected),len(rows)),
        'accepted_identity_accuracy':proportion(right,len(accepted)),'accuracy_95ci':wilson(right,len(accepted)),
        'silent_failure_rate':proportion(len(accepted)-right,len(accepted)),
        'rejection_precision':proportion(sum(not r['correct'] for r in rejected),len(rejected)),
        'error_rejection_recall':proportion(sum(not r['correct'] for r in rejected),len(wrong)),
        'source_field_accuracy':proportion(sum(fields),len(fields)),
        'unknown_false_acceptance':proportion(sum(r['kind']!='known' for r in accepted),sum(r['kind']!='known' for r in rows)),
        'mean_seconds':statistics.mean(r['seconds'] for r in rows),'p95_seconds':float(np.percentile([r['seconds'] for r in rows],95))}


def run():
    manifest=json.loads((ART/'manifest.json').read_text());protocol=json.loads((ART/'protocol.json').read_text()); search=VisualSearch()
    if search.meta['protocol_hash']!=digest(protocol):raise ValueError('Protocol changed; rebuild index')
    ids=[r['source_id'] for r in manifest];sha=[r['sha256'] for r in manifest]
    if len(ids)!=len(set(ids)) or len(sha)!=len(set(sha)):raise ValueError('Duplicate query identity/image across splits')
    if len(manifest)!=100 or sum(r['split']=='test' for r in manifest)!=60:raise ValueError('Unexpected experiment size')
    if set(search.meta['image_hashes']) & set(protocol['excluded_image_hashes']):raise ValueError('Unknown image leaked into gallery')
    baseline_started=time.perf_counter();baseline=HandcraftedBaseline(search.rows,GALLERY)
    baseline_index_seconds=round(time.perf_counter()-baseline_started,3)
    results=[]
    for i,row in enumerate(manifest):
        for variant,path in [('transformed',row['image']),('original',row['original'])]:
            image=read_image(ROOT/path)
            started=time.perf_counter();vector=search.encoder.encode([image])[0];domain=search.encoder.domain(vector);raw_hits=search.rank(vector,k=100)
            hits=search.verifier.rerank(image,raw_hits)
            secs=time.perf_counter()-started
            top=hits[0]
            results.append({**row,'variant':variant,'method':'clip','predicted_id':top['record']['id'],'predicted_facts':top['record'],
               'score':top['score'],'margin':round(top['score']-hits[1]['score'],6),'correct':row['expected_id']==top['record']['id'],
               'is_footwear':domain['is_footwear'],'domain':domain,'top3':[h['record']['id'] for h in hits[:3]],'seconds':secs,
               'clip_score':top.get('clip_score',top['score']),'geometric_inliers':top.get('geometric_inliers')})
            baseline_started=time.perf_counter();simple_hits=baseline.rank(image,k=5);baseline_seconds=time.perf_counter()-baseline_started
            simple_top=simple_hits[0]
            results.append({**row,'variant':variant,'method':'handcrafted','predicted_id':simple_top['record']['id'],
               'predicted_facts':simple_top['record'],'score':simple_top['score'],
               'margin':round(simple_top['score']-simple_hits[1]['score'],6),
               'correct':row['expected_id']==simple_top['record']['id'],
               'is_footwear':True,'domain':None,'top3':[h['record']['id'] for h in simple_hits[:3]],
               'seconds':baseline_seconds})
        if (i+1)%10==0:print(f'Evaluated {i+1}/100',flush=True)
    save(ART/'predictions.json',{'manifest_hash':digest(manifest),'index_signature':search.signature,'rows':results})
    report={'protocol':protocol,'index_size':len(search.rows),'model':search.meta['model'],'device':search.encoder.device,
      'signature':search.signature,'manifest_hash':digest(manifest),'results':{},'controls':{},'llm_evaluated':False,'main_method':'CLIP top100 + SIFT affine RANSAC reranking',
      'baseline_method':'0.6 dHash similarity + 0.4 foreground RGB histogram intersection, no trained model',
      'baseline_index_seconds':baseline_index_seconds,
      'api_cost_usd':0,'local_hardware_cost_usd':None,'business_time_reduction':None,
      'limitations':['Queries are deterministic transforms of catalog photos in a closed-gallery experiment.','The handcrafted baseline has no footwear classifier; it uses the same score-and-margin calibration gate.','No user study or paid OpenRouter calls were run.','60 held-out queries: uncertainty is substantial.','CLIP pretraining overlap with this public dataset is unknown.']}
    curve=[]
    for method in ('clip','handcrafted'):
        calibr=[r for r in results if r['split']=='calibration' and r['variant']=='transformed' and r['method']==method]
        cal=select_threshold(calibr,target=.85,min_accepted=10)
        definition=('Score: 0.7 * min(affine_inliers/20,1) * inliers/ratio_matches + 0.3 * CLIP cosine; top1-top2 margin.'
                    if method=='clip' else 'Score: 0.6 * dHash bit similarity + 0.4 * foreground RGB histogram intersection; top1-top2 margin.')
        cal.update(signature=search.signature,method=method,n=len(calibr),target=.85,manifest_hash=digest(manifest),
                   definition=definition+' No score is a probability.')
        save(ART/f'{method}-calibration.json',cal)
        if method=='clip':save(ART/'calibration.json',cal)
        tests=[r for r in results if r['split']=='test' and r['variant']=='transformed' and r['method']==method]
        report['results'][method]={'calibration':cal,'test':metrics(tests,cal)}
        originals=[r for r in results if r['split']=='test' and r['variant']=='original' and r['method']==method]
        report['controls'][method]={'original':metrics(originals,cal),'transformed':metrics(tests,cal)}
        for t in sorted({r['score'] for r in calibr}):
            m=metrics(calibr,{**cal,'threshold':t})
            curve.append({'method':method,'split':'calibration','threshold':t,'margin':cal['margin'],'coverage':m['coverage'],'precision':m['accepted_identity_accuracy']})
    save(ART/'evaluation.json',report);save(ART/'calibration-curve.json',curve)
    # Failure list intentionally retains all test failures, not selected successes.
    cal=report['results']['clip']['calibration'];failures=[]
    for r in results:
        if r['split']=='test' and r['variant']=='transformed' and r['method']=='clip':
            accepted=r['is_footwear'] and r['score']>=cal['threshold'] and r['margin']>=cal['margin']
            if not r['correct'] or not accepted:failures.append({**r,'accepted':accepted})
    save(ART/'failure-cases.json',failures)
    print(json.dumps(report['results'],indent=2),flush=True)


def study(path):
    return analyze_user_study(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path);a=p.parse_args()
    study(a.study) if a.study else run()
