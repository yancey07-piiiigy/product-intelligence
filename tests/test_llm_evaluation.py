"""Regression tests for source-grounded summary evaluation and cost reporting."""
import unittest

from evaluate_summary import summarize_metrics
from summary_service import billed_cost
from server import cost_breakdown


class EvaluationAccountingTests(unittest.TestCase):
    def test_provider_bill_takes_precedence_over_estimated_rates(self):
        amount, source = billed_cost({'cost': 0.0023, 'prompt_tokens': 100}, {'input_tokens': 100})
        self.assertEqual((amount, source), (0.0023, 'provider_reported'))

    def test_missing_price_is_unknown(self):
        amount, source = billed_cost({'prompt_tokens': 100}, {'input_tokens': 100})
        self.assertIsNone(amount)
        self.assertEqual(source, 'unknown')

    def test_wrong_identity_counts_against_end_to_end_claim_accuracy(self):
        rows = [
            {'expected_id': '1', 'predicted_id': '1', 'ground_truth': {'articleType': 'Sports Shoes'},
             'summary_source': {'articleType': 'Sports Shoes'}, 'retrieval_seconds': 0.1, 'summary_wall_seconds': 0.2,
             'summary': {'mode': 'openrouter_grounded', 'claims': [{'field': 'articleType', 'value': 'Sports Shoes'}], 'api_usd': 0.001, 'cost_source': 'provider_reported', 'cached': False}},
            {'expected_id': '2', 'predicted_id': '3', 'ground_truth': {'articleType': 'Flip Flops'},
             'summary_source': {'articleType': 'Sports Shoes'}, 'retrieval_seconds': 0.1, 'summary_wall_seconds': 0.2,
             'summary': {'mode': 'openrouter_grounded', 'claims': [{'field': 'articleType', 'value': 'Sports Shoes'}], 'api_usd': None, 'cost_source': 'unknown', 'cached': False}},
        ]
        result = summarize_metrics(rows, 2)
        self.assertEqual(result['source_supported_claim_rate'], 1.0)
        self.assertEqual(result['end_to_end_claim_accuracy'], 0.5)
        self.assertIsNone(result['total_api_usd_this_run'])
        self.assertEqual(result['cost_known_calls'], 1)

    def test_single_search_total_is_unknown_without_local_rate(self):
        amount = cost_breakdown(0.05, 0.002, None)
        self.assertEqual(amount['api_usd'], 0.002)
        self.assertIsNone(amount['total_estimated_usd'])
        self.assertEqual(cost_breakdown(0.05, 0.002, 0.0001)['total_estimated_usd'], 0.0021)

    def test_full_set_cost_includes_original_cost_of_cache_hits(self):
        rows = [
            {'expected_id': '1', 'predicted_id': '1', 'ground_truth': {'articleType': 'Shoes'},
             'summary_source': {'articleType': 'Shoes'}, 'retrieval_seconds': .05, 'summary_wall_seconds': .1,
             'summary': {'mode': 'openrouter_grounded', 'claims': [{'field': 'articleType', 'value': 'Shoes'}],
                         'api_usd': 0, 'original_api_usd': .002, 'cost_source': 'cache_hit', 'cached': True}},
            {'expected_id': '2', 'predicted_id': '2', 'ground_truth': {'articleType': 'Shoes'},
             'summary_source': {'articleType': 'Shoes'}, 'retrieval_seconds': .05, 'summary_wall_seconds': .3,
             'summary': {'mode': 'openrouter_grounded', 'claims': [{'field': 'articleType', 'value': 'Shoes'}],
                         'api_usd': .003, 'cost_source': 'provider_reported', 'cached': False}},
        ]
        result = summarize_metrics(rows, 2, 2)
        self.assertEqual(result['total_api_usd_this_run'], .003)
        self.assertEqual(result['total_api_usd_original_generation'], .005)
        self.assertEqual(result['mean_api_usd_per_accepted_image_original_generation'], .0025)


if __name__ == '__main__':
    unittest.main()
