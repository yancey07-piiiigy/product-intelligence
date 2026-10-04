import io,json,tempfile,unittest,os,ssl
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from summary_service import validate_claims,summarize,cost
from vision import read_image,choose,canonical
from benchmark import metrics

RECORD={'id':'fixture','articleType':'Sports Shoes','baseColour':'Blue','usage':'Sports','gender':'Unisex','season':'Summer'}
class SafetyTests(unittest.TestCase):
    def test_no_key_has_explicit_template_mode(self):
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':''}):r=summarize(RECORD)
        self.assertEqual(r['mode'],'template');self.assertTrue(r['warning']);self.assertEqual(r['api_usd'],0)
    def test_wrong_source_rejected(self):
        with self.assertRaises(ValueError):validate_claims({'source_id':'other','claims':[]},RECORD)
    def test_wrong_value_rejected(self):
        with self.assertRaises(ValueError):validate_claims({'source_id':'fixture','claims':[{'field':'baseColour','value':'Red'}]},RECORD)
    def test_missing_core_claim_rejected(self):
        with self.assertRaises(ValueError):validate_claims({'source_id':'fixture','claims':[{'field':'articleType','value':'Sports Shoes'}]},RECORD)
    def test_free_brand_claim_rejected(self):
        with self.assertRaises(ValueError):validate_claims({'source_id':'fixture','claims':[{'field':'brand','value':'Nike'}]},RECORD)
    def test_paid_response_uses_validated_fields_and_usage(self):
        selected={'source_id':'fixture','claims':[{'field':k,'value':RECORD[k]} for k in ('articleType','baseColour','usage')]}
        payload={'choices':[{'message':{'content':json.dumps(selected)}}],'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}}
        class Response(io.BytesIO):pass
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'artifacts').mkdir()
            with patch('summary_service.ROOT',p),patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-key','OPENROUTER_INPUT_USD_PER_M':'1','OPENROUTER_OUTPUT_USD_PER_M':'2'}),patch('urllib.request.urlopen',return_value=Response(json.dumps(payload).encode())) as call:
                r=summarize(RECORD);cached=summarize(RECORD)
                self.assertEqual(call.call_count,1);self.assertEqual(r['mode'],'openrouter_grounded');self.assertAlmostEqual(r['api_usd'],.0002);self.assertTrue(cached['cached']);self.assertEqual(cached['api_usd'],0)
                request=call.call_args.args[0];body=json.loads(request.data)
                self.assertEqual(request.full_url,'https://openrouter.ai/api/v1/chat/completions')
                self.assertEqual(body['response_format']['type'],'json_schema')
                self.assertTrue(body['provider']['require_parameters'])
                self.assertEqual(call.call_args.kwargs['context'].verify_mode,ssl.CERT_REQUIRED)
    def test_bad_image_rejected(self):
        with self.assertRaises(ValueError):read_image(b'not an image')
    def test_nonfootwear_never_accepted(self):
        self.assertFalse(choose([],{'is_footwear':False},None,'')[0])
    def test_stale_index_never_accepted(self):
        self.assertFalse(choose([],{'is_footwear':True},{'signature':'old'},'new')[0])
    def test_zero_margin_never_passes_positive_threshold(self):
        h=[{'score':.99},{'score':.99}];c={'signature':'s','threshold':.8,'margin':.001}
        self.assertFalse(choose(h,{'is_footwear':True},c,'s')[0])
    def test_rejection_metrics_include_wrong_unknowns(self):
        base={'kind':'known','expected_id':'1','top3':['1'],'predicted_facts':{},'ground_truth':{},'seconds':.1,'is_footwear':True,'margin':.02}
        rows=[{**base,'correct':True,'score':.95},{**base,'correct':False,'score':.3,'kind':'unknown_footwear'}]
        m=metrics(rows,{'threshold':.9,'margin':.01})
        self.assertEqual(m['coverage'],.5);self.assertEqual(m['rejection_precision'],1);self.assertEqual(m['error_rejection_recall'],1)
    def test_no_acceptance_has_undefined_accuracy(self):
        row={'kind':'non_footwear','expected_id':None,'top3':['1'],'predicted_facts':{},'ground_truth':{},'seconds':.1,'is_footwear':False,'correct':False,'score':.3,'margin':0}
        self.assertIsNone(metrics([row],{'threshold':1.1,'margin':1.1})['accepted_identity_accuracy'])
if __name__=='__main__':unittest.main()
