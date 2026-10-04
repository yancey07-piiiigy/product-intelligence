"""Exercise the analyze endpoint without model weights, sockets or paid calls."""
import base64
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import server

class VisualAPITests(unittest.TestCase):
    def test_html_shell_is_not_cached_between_frontend_builds(self):
        handler=object.__new__(server.Handler)
        headers=[]
        handler.send_response=lambda status: None
        handler.send_header=lambda key,value: headers.append((key,value))
        handler.end_headers=lambda: None
        handler.wfile=io.BytesIO()
        handler.send(200,b'<html></html>','text/html')
        self.assertIn(('Cache-Control','no-store'),headers)

    def test_unknown_api_path_returns_json_error_not_frontend_html(self):
        handler=object.__new__(server.Handler)
        handler.path='/api/unknown-version'
        replies=[]
        handler.json=lambda body,status=200: replies.append((status,body))
        handler.do_GET()
        self.assertEqual(replies,[(404,{'error':'API route not found. Restart the Python service if the frontend was updated.'})])

    def test_uncertain_observation_preserves_status_and_logs_usage(self):
        result={'accepted':False,'status':'uncertain','candidates':[], 'score':.5,'margin':.001,'local_seconds':.1,'api_usd':0}
        visual={'text':'白色系带鞋','mode':'visual_observation','scope':'uploaded_image','api_usd':.002,'usage':{'total_tokens':60}}
        payload=json.dumps({'image':base64.b64encode(b'image').decode()}).encode()
        handler=object.__new__(server.Handler)
        handler.path='/api/analyze'; handler.headers={'Content-Length':str(len(payload))}
        handler.client_address=('test-client',0);handler.rfile=io.BytesIO(payload)
        responses=[];handler.json=lambda body,status=200: responses.append((status,body))
        search=SimpleNamespace(search=lambda raw:dict(result))
        with patch.object(server,'search',search,create=True),patch.object(server,'maybe_describe',return_value=visual) as describe,patch.object(server,'log_event') as log,patch.object(server,'local_cost',return_value=None):
            handler.do_POST()
        self.assertEqual(responses[0][0],200)
        body=responses[0][1]
        self.assertFalse(body['accepted']);self.assertEqual(body['status'],'uncertain')
        self.assertIsNone(body['summary']);self.assertEqual(body['visual_summary'],visual)
        self.assertIs(describe.call_args.args[2],True)
        self.assertEqual(log.call_args.args[0]['usage'],{'total_tokens':60})
        self.assertEqual(log.call_args.args[0]['summary_mode'],'visual_observation')
        self.assertEqual(log.call_args.args[0]['api_usd'],.002)
