import io,json,os,ssl,unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch
from PIL import Image
from vision_summary import describe_image,maybe_describe
class VisionSummaryTests(unittest.TestCase):
    def test_gate(self):
        with patch('vision_summary.describe_image') as call:
            for status in ('accepted','out_of_scope','uncalibrated'):
                self.assertIsNone(maybe_describe({'status':status},b'',True))
            self.assertIsNone(maybe_describe({'status':'uncertain'},b'',False))
            call.assert_not_called()
            maybe_describe({'status':'uncertain'},b'',True)
            call.assert_called_once()
    def test_image_payload_and_observation(self):
        raw=io.BytesIO();Image.new('RGB',(20,20)).save(raw,format='PNG')
        content={'is_footwear':True,'shoe_type':'低帮鞋','colours':'白色','closure':'系带','visible_details':'深色鞋底'}
        response={'choices':[{'message':{'content':json.dumps(content)}}],'usage':{'prompt_tokens':30,'completion_tokens':20,'total_tokens':50,'cost':.001}}
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}),patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result=describe_image(raw.getvalue())
        self.assertFalse(result['identity_confirmed']);self.assertIn('白色',result['text'])
        payload=json.loads(call.call_args.args[0].data)
        self.assertTrue(payload['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(call.call_args.kwargs['context'].verify_mode,ssl.CERT_REQUIRED)
        self.assertNotIn('source_id',result)
    def test_missing_key(self):
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':''}):
            self.assertIsNone(describe_image(b'')['text'])
    def test_failure(self):
        raw=io.BytesIO();Image.new('RGB',(20,20)).save(raw,format='PNG')
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}),patch('urllib.request.urlopen',side_effect=TimeoutError):
            result=describe_image(raw.getvalue())
        self.assertIsNone(result['text']);self.assertTrue(result['warning'])
    def test_http_failures_have_actionable_safe_messages(self):
        raw=io.BytesIO();Image.new('RGB',(20,20)).save(raw,format='PNG')
        cases={400:'request',401:'API key',402:'credits',403:'permission',404:'model',429:'rate limit',503:'provider'}
        for status,expected in cases.items():
            error=HTTPError('https://openrouter.ai/api/v1/chat/completions',status,'failure',{},io.BytesIO(b'{"error":{"message":"secret-test-key"}}'))
            with self.subTest(status=status),patch.dict(os.environ,{'OPENROUTER_API_KEY':'secret-test-key'}),patch('urllib.request.urlopen',side_effect=error):
                result=describe_image(raw.getvalue())
            self.assertIn(expected,result['warning'])
            self.assertNotIn('secret-test-key',result['warning'])
            self.assertEqual(result['failure_reason'],f'http_{status}')
            self.assertIsNone(result['api_usd'])
    def test_network_failure_is_distinct_from_bad_response(self):
        raw=io.BytesIO();Image.new('RGB',(20,20)).save(raw,format='PNG')
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}),patch('urllib.request.urlopen',side_effect=URLError('offline')):
            result=describe_image(raw.getvalue())
        self.assertEqual(result['failure_reason'],'network')
        self.assertIn('network',result['warning'])
    def test_malformed_envelope_falls_back(self):
        raw=io.BytesIO();Image.new('RGB',(20,20)).save(raw,format='PNG')
        for response in ([], {'usage':None}, {'usage':'invalid'}):
            with self.subTest(response=response), patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
                result=describe_image(raw.getvalue())
                self.assertIsNone(result['text'])
                self.assertTrue(result['warning'])
                self.assertIsNone(result['api_usd'])

    def test_invalid_observation_retains_billed_cost(self):
        raw=io.BytesIO();Image.new('RGB',(20,20)).save(raw,format='PNG')
        for content in ({'is_footwear':True,'brand':'invented'}, {'is_footwear':False,'shoe_type':'未知','colours':'未知','closure':'未知','visible_details':'未知'}):
            response={'choices':[{'message':{'content':json.dumps(content)}}],'usage':{'cost':.001}}
            with self.subTest(content=content), patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}),patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
                result=describe_image(raw.getvalue())
                self.assertIsNone(result['text'])
                self.assertEqual(result['api_usd'],.001)
