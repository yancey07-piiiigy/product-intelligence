"""Evaluate source-grounded OpenRouter summaries on frozen, accepted test queries.

Live requests are opt-in and capped with --max-summaries. This measures catalog
claim fidelity, not independently judged quality of free-form visual observations.
"""
import argparse
import json
import os
import time

from core import FIELDS
from prepare_project import save
from summary_service import load_env, summarize
from vision import ART


def summarize_metrics(rows, test_n, accepted_n=None):
    """Keep source fidelity, end-to-end correctness, and billing denominators separate."""
    generated = [r for r in rows if r['summary']['mode'] == 'openrouter_grounded']
    claims = [(r, c) for r in generated for c in r['summary']['claims']]
    source_matches = sum(c['value'] == r['summary_source'].get(c['field']) for r, c in claims)
    truth_matches = sum(r['expected_id'] == r['predicted_id'] and c['value'] == r['ground_truth'].get(c['field']) for r, c in claims)
    billed = [r for r in rows if not r['summary'].get('cached') and r['summary'].get('cost_source') != 'no_api_call']
    known = [r for r in billed if r['summary'].get('api_usd') is not None]
    complete = accepted_n is not None and len(rows) == accepted_n
    total = sum(r['summary']['api_usd'] for r in known) if len(known) == len(billed) else None
    original_costs = [r['summary'].get('original_api_usd') if r['summary'].get('cached')
                      else r['summary'].get('api_usd') for r in rows]
    original_total = sum(original_costs) if all(value is not None for value in original_costs) else None
    return {
        'n_test_queries': test_n,
        'n_accepted_expected': accepted_n,
        'n_evaluated': len(rows),
        'complete_accepted_set': complete,
        'n_openrouter_validated': len(generated),
        'n_template_fallbacks': len(rows) - len(generated),
        'openrouter_generation_success_rate': len(generated) / len(rows) if rows else None,
        'source_supported_claim_rate': source_matches / len(claims) if claims else None,
        'end_to_end_claim_accuracy': truth_matches / len(claims) if claims else None,
        'emitted_claims': len(claims),
        'cache_hits': sum(bool(r['summary'].get('cached')) for r in rows),
        'billed_calls': len(billed),
        'cost_known_calls': len(known),
        'cost_unknown_calls': len(billed) - len(known),
        'provider_reported_cost_calls': sum(r['summary'].get('cost_source') == 'provider_reported' for r in billed),
        'token_rate_estimated_cost_calls': sum(r['summary'].get('cost_source') == 'configured_rate_estimate' for r in billed),
        'total_api_usd_this_run': total,
        'mean_api_usd_per_test_image_this_run': total / test_n if complete and total is not None and test_n else None,
        'total_api_usd_original_generation': original_total,
        'mean_api_usd_per_accepted_image_original_generation': original_total / accepted_n if complete and original_total is not None and accepted_n else None,
        'original_cost_unknown_calls': sum(value is None for value in original_costs),
        'mean_retrieval_plus_summary_seconds': (sum(r['retrieval_seconds'] + r['summary_wall_seconds'] for r in rows) / len(rows)) if rows else None,
        'limitations': 'Claims are checked against catalog fields and frozen query IDs. Free-form visual observations and user understanding are not manually rated. Costs exclude local electricity/hardware unless separately estimated.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-summaries', type=int, default=0,
                        help='Maximum accepted queries to process; limits possible paid calls. 0 makes no requests.')
    args = parser.parse_args()
    if args.max_summaries <= 0:
        parser.error('Set --max-summaries N after deciding the maximum number of possible paid calls.')
    load_env()
    if not os.getenv('OPENROUTER_API_KEY'):
        parser.error('Configure OPENROUTER_API_KEY in .env first; no calls were made.')

    report = json.loads((ART / 'evaluation.json').read_text())
    calibration = report['results']['clip']['calibration']
    predictions = json.loads((ART / 'predictions.json').read_text())
    if predictions['index_signature'] != calibration['signature']:
        parser.error('Stale calibration/predictions; rerun the benchmark.')
    test_rows = [r for r in predictions['rows'] if r['method'] == 'clip' and r['variant'] == 'transformed' and r['split'] == 'test']
    accepted = [r for r in test_rows if r['is_footwear'] and r['score'] >= calibration['threshold'] and r['margin'] >= calibration['margin']]
    results = []
    for row in accepted[:args.max_summaries]:
        started = time.perf_counter()
        summary = summarize(row['predicted_facts'])
        results.append({
            'query_id': row['query_id'], 'expected_id': row['expected_id'], 'predicted_id': row['predicted_id'],
            'ground_truth': row['ground_truth'],
            'summary_source': {k: row['predicted_facts'].get(k) for k in FIELDS},
            'summary': summary, 'retrieval_seconds': row['seconds'],
            'summary_wall_seconds': time.perf_counter() - started,
        })
        results[-1]['end_to_end_seconds'] = results[-1]['retrieval_seconds'] + results[-1]['summary_wall_seconds']
        save(ART / 'llm-predictions.json', results)
        print(row['query_id'], summary['mode'], 'cached=' + str(summary['cached']), flush=True)

    summary_report = summarize_metrics(results, len(test_rows), len(accepted))
    summary_report['model'] = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.5-flash-lite')
    summary_report['sample_selection'] = (
        'All accepted queries in the frozen held-out set.' if summary_report['complete_accepted_set']
        else 'First N accepted queries in frozen manifest order; a partial run is not a representative accuracy estimate.')
    save(ART / 'llm-evaluation.json', summary_report)
    report['llm_evaluated'] = bool(results)
    report['llm_summary'] = summary_report
    save(ART / 'evaluation.json', report)
    print(json.dumps(summary_report, indent=2))


if __name__ == '__main__':
    main()
