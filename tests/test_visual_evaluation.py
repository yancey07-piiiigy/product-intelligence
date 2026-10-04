"""Cost and review accounting for uncertain-shoe descriptions."""
import unittest

from evaluate_visual import visual_metrics


class VisualEvaluationTests(unittest.TestCase):
    def test_complete_set_keeps_cost_and_human_review_separate(self):
        rows = [
            {'query_id': 'q1', 'wall_seconds': 1.2, 'reused': True,
             'observation': {'text': 'Shoe type: dress shoes.', 'api_usd': .002, 'api_seconds': 1.1}},
            {'query_id': 'q2', 'wall_seconds': 1.5, 'reused': False,
             'observation': {'text': 'Shoe type: sandals.', 'api_usd': .003, 'api_seconds': 1.4}},
        ]
        reviews = {'q1': {'shoe_type_correct': 'yes', 'colours_correct': 'yes',
                          'closure_correct': 'yes', 'visible_details_correct': 'yes',
                          'unsupported_claims': 'no'}}
        result = visual_metrics(rows, expected_n=2, reviews=reviews)
        self.assertEqual(result['total_api_usd_original_generation'], .005)
        self.assertEqual(result['total_api_usd_this_run'], .003)
        self.assertEqual(result['mean_api_seconds'], 1.25)
        self.assertEqual(result['human_review_complete'], 1)
        self.assertIsNone(result['human_review_field_accuracy'])


if __name__ == '__main__':
    unittest.main()
