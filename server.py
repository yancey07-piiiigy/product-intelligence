"""React + Python local application. No user images or API keys are persisted."""
import argparse,base64,json,mimetypes,os,threading,time,uuid
from collections import defaultdict,deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from core import ROOT,FIELDS
from vision import VisualSearch,ART,GALLERY
from summary_service import summarize,load_env
from vision_summary import maybe_describe
from user_study import load_tasks,public_manual_page

lock=threading.Lock(); sessions={}; rates=defaultdict(deque)
search=None  # Initialized once when the local server starts.


def log_event(event):
    with (ART/'runtime.jsonl').open('a') as f:f.write(json.dumps({'timestamp':time.time(),**event},ensure_ascii=False)+'\n')


def local_cost(seconds):
    rate=os.getenv('LOCAL_COMPUTE_USD_PER_HOUR')
    if not rate:return None
    value=float(rate)
    if value<0:raise ValueError('LOCAL_COMPUTE_USD_PER_HOUR must be nonnegative')
    return round(seconds*value/3600,8)


def cost_breakdown(local_seconds,api_usd,local_estimated_usd,api_cost_source='no_api_call'):
    """Separate provider billing from optional local compute estimate."""
    return {'local_seconds':local_seconds,'api_usd':api_usd,'api_cost_source':api_cost_source,
            'local_estimated_usd':local_estimated_usd,
            'total_estimated_usd':round(api_usd+local_estimated_usd,8) if api_usd is not None and local_estimated_usd is not None else None}


class Handler(BaseHTTPRequestHandler):
    def send(self,status,body,content='application/json; charset=utf-8'):
        self.send_response(status);self.send_header('Content-Type',content);self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store' if content.startswith(('application/json','text/html')) else 'public,max-age=300');self.end_headers();self.wfile.write(body)
    def json(self,obj,status=200):self.send(status,json.dumps(obj,ensure_ascii=False).encode())
    def do_GET(self):
        parsed=urlparse(self.path);path=parsed.path;params=parse_qs(parsed.query)
        if path=='/api/status':
            subset=json.loads((ROOT/'data/gallery-subset.json').read_text())
            return self.json({'index_size':len(search.rows),'source_footwear_size':subset['source_available_footwear'],'sampling_strategy':subset['strategy'],'model':search.meta['model'],'device':search.encoder.device,'index_seconds':search.meta['build_seconds']+json.loads((ART/'geometry.json').read_text())['seconds'],
               'calibrated':bool(search.calibration and search.calibration.get('signature')==search.signature),'calibration':search.calibration,
               'api_key_configured':bool(os.getenv('OPENROUTER_API_KEY')),'summary_model':os.getenv('OPENROUTER_MODEL','google/gemini-2.5-flash-lite'),
               'vision_model':os.getenv('OPENROUTER_VISION_MODEL') or 'google/gemini-2.5-flash-lite',
               'rates_configured':bool(os.getenv('OPENROUTER_INPUT_USD_PER_M') and os.getenv('OPENROUTER_OUTPUT_USD_PER_M'))})
        if path=='/api/evaluation':
            p=ART/'evaluation.json';return self.json(json.loads(p.read_text()) if p.exists() else {'pending':True})
        if path=='/api/study/results':
            p=ART/'user-study-results.json'
            return self.json(json.loads(p.read_text()) if p.exists() else {'pending':True})
        if path=='/api/failures':
            p=ART/'failure-cases.json';return self.json(json.loads(p.read_text()) if p.exists() else [])
        if path=='/api/study/manual':
            task_id=params.get('task',[''])[0]
            task=next((t for t in load_tasks() if t['task_id']==task_id),None)
            if task is None:return self.json({'error':'Unknown study task'},404)
            try:page=int(params.get('page',['1'])[0])
            except ValueError:page=1
            return self.json(public_manual_page(task,search.rows,page))
        if path=='/api/catalog':
            q=params.get('q',[''])[0].lower();category=params.get('category',[''])[0]
            rows=[r for r in search.rows if (not category or r['articleType']==category) and (not q or q in ' '.join(r.values()).lower())]
            try:page=max(1,int(params.get('page',['1'])[0]))
            except ValueError:page=1
            return self.json({'total':len(rows),'page':page,'page_size':24,'categories':sorted({r['articleType'] for r in search.rows}),'items':rows[(page-1)*24:page*24]})
        if path.startswith('/api/'):
            return self.json({'error':'API route not found. Restart the Python service if the frontend was updated.'},404)
        if path.startswith('/image/'):
            pid=path.split('/')[-1]
            if pid not in source_ids:return self.json({'error':'Image not found'},404)
            file=GALLERY/f'{pid}.jpg'
        elif path.startswith('/query/'):
            row=queries.get(path.split('/')[-1])
            if row is None:return self.json({'error':'Query not found'},404)
            file=ROOT/row['image']
        else:
            rel=path.lstrip('/') or 'index.html';file=(ROOT/'frontend/dist'/rel).resolve();dist=(ROOT/'frontend/dist').resolve()
            if not file.is_relative_to(dist):return self.json({'error':'Not found'},404)
            if not file.is_file():
                if '.' not in rel:file=dist/'index.html'
                else:return self.json({'error':'Not found'},404)
        if not file.is_file():return self.json({'error':'File missing. Build frontend first.'},404)
        return self.send(200,file.read_bytes(),mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
    def do_POST(self):
        if self.path not in ('/api/analyze','/api/candidate-summary'):return self.json({'error':'Not found'},404)
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}','http://127.0.0.1:5173','http://localhost:5173'):return self.json({'error':'Origin not allowed'},403)
        try:
            now=time.monotonic(); queue=rates[self.client_address[0]]
            while queue and queue[0]<now-60:queue.popleft()
            if len(queue)>=20:return self.json({'error':'Limit: 20 requests per minute. Please try again shortly.'},429)
            queue.append(now)
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>12_000_000:raise ValueError('Upload limit: 8 MB')
            p=json.loads(self.rfile.read(size))
            with lock:
                if self.path=='/api/analyze':
                    raw=base64.b64decode(p.get('image',''),validate=True)
                    if not raw or len(raw)>8_000_000:raise ValueError('Choose an image smaller than 8 MB')
                    result=search.search(raw);sid=uuid.uuid4().hex
                    sessions[sid]={'at':now,'candidates':result['candidates']}
                    for old in list(sessions):
                        if sessions[old]['at']<now-1800:del sessions[old]
                    result.update(session_id=sid,summary=None,summary_scope='matched_record',local_estimated_usd=local_cost(result['local_seconds']))
                    if result['accepted']:result['summary']=summarize(result['candidates'][0]['record'],p.get('use_llm',True))
                    result['visual_summary']=maybe_describe(result,raw,True)
                    if result['visual_summary']:result['api_usd']=result['visual_summary']['api_usd']
                    if result['summary']:result['api_usd']=result['summary']['api_usd']
                    active_summary=result['summary'] or result['visual_summary']
                    result['cost_breakdown']=cost_breakdown(result['local_seconds'],result['api_usd'],result['local_estimated_usd'],
                                                            active_summary.get('cost_source','unknown') if active_summary else 'no_api_call')
                    log_event({'event':'analyze','status':result['status'],'score':result['score'],'margin':result['margin'],'local_seconds':result['local_seconds'],'api_usd':result['api_usd'],'cost_breakdown':result['cost_breakdown'],'summary_mode':active_summary['mode'] if active_summary else None,'usage':active_summary.get('usage') if active_summary else None})
                    return self.json(result)
                session=sessions.get(p.get('session_id'))
                if not session or session['at']<now-1800:raise ValueError('This search session has expired. Please upload the image again.')
                row=next((h['record'] for h in session['candidates'] if h['record']['id']==p.get('product_id')),None)
                if row is None:raise ValueError('Select a candidate from this search.')
                result=summarize(row,p.get('use_llm',True));result['scope']='user_selected_candidate'
                log_event({'event':'candidate_summary','source_id':row['id'],'api_usd':result['api_usd'],'usage':result.get('usage'),'mode':result['mode']})
                return self.json(result)
        except (ValueError,KeyError,TypeError) as e:return self.json({'error':str(e)},400)
        except Exception as e:return self.json({'error':f'Processing failed ({type(e).__name__}). Check the server logs and model files.'},500)
    def log_message(self,*args):pass

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8501);args=parser.parse_args();load_env()
    if not (ROOT/'frontend/dist/index.html').exists():parser.error('Run cd frontend && npm ci && npm run build first')
    search=VisualSearch();manifest=json.loads((ART/'manifest.json').read_text());queries={r['query_id']:r for r in manifest};source_ids={r['id'] for r in search.rows}|{r['source_id'] for r in manifest}
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'ShoeLens React + CLIP ready: http://127.0.0.1:{args.port} | {len(search.rows)} products',flush=True);server.serve_forever()
