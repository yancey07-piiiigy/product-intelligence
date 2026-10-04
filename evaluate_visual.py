"""Evaluate OpenRouter observations on every held-out uncertain footwear image.

The script attempts at most --max-images new paid calls, checkpoints each result,
and writes a CSV for a human reviewer. Run again without --max-images after
filling the review columns to refresh the human-review metrics without API calls.
"""
import argparse
import base64
import csv
import hashlib
import html
import json
import os
import statistics
import time
from pathlib import Path

from summary_service import load_env
from vision import ART, ROOT
from vision_summary import describe_image

PREDICTIONS = ART / 'predictions.json'
RESULTS = ART / 'visual-predictions.json'
REPORT = ART / 'visual-evaluation.json'
REVIEWS = ART / 'visual-human-review.csv'
REVIEW_HTML = ART / 'visual-human-review.html'
REVIEW_FIELDS = ('shoe_type_correct', 'colours_correct', 'closure_correct',
                 'visible_details_correct', 'unsupported_claims')
CSV_FIELDS = ('query_id', 'image', 'description', *REVIEW_FIELDS, 'notes')


def visual_metrics(rows, expected_n, reviews):
    current = [r['observation'].get('api_usd') for r in rows if not r.get('reused')]
    original = [r['observation'].get('api_usd') for r in rows]
    api_times = [r['observation']['api_seconds'] for r in rows if r['observation'].get('api_seconds') is not None]
    reviewed = [reviews.get(r['query_id'], {}) for r in rows]
    complete_reviews = [r for r in reviewed if all(r.get(k) in ('yes', 'no') for k in REVIEW_FIELDS)]
    all_reviewed = len(rows) == expected_n and len(complete_reviews) == expected_n
    field_values = [r[k] for r in complete_reviews for k in REVIEW_FIELDS[:4]]
    return {
        'expected_uncertain_footwear': expected_n,
        'n_evaluated': len(rows),
        'complete_uncertain_set': len(rows) == expected_n,
        'vision_descriptions_generated': sum(bool(r['observation'].get('text')) for r in rows),
        'vision_failures': sum(not r['observation'].get('text') for r in rows),
        'reused_results': sum(bool(r.get('reused')) for r in rows),
        'new_calls_this_run': sum(not r.get('reused') for r in rows),
        'total_api_usd_this_run': sum(current) if all(v is not None for v in current) else None,
        'total_api_usd_original_generation': sum(original) if all(v is not None for v in original) else None,
        'original_cost_unknown_calls': sum(v is None for v in original),
        'mean_api_seconds': statistics.mean(api_times) if api_times else None,
        'mean_wall_seconds': statistics.mean(r['wall_seconds'] for r in rows) if rows else None,
        'mean_retrieval_plus_visual_seconds': statistics.mean(r.get('retrieval_seconds', 0) + r['wall_seconds'] for r in rows) if rows else None,
        'human_review_complete': len(complete_reviews),
        'human_review_field_accuracy': (sum(v == 'yes' for v in field_values) / len(field_values)) if all_reviewed and field_values else None,
        'human_review_unsupported_claims': sum(r['unsupported_claims'] == 'yes' for r in complete_reviews) if all_reviewed else None,
        'limitations': 'Review fields require a person to inspect each image and description; blank fields are not scored. Photos are transformed catalog images, not independent phone photos.',
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_review_html(rows, reviews):
    cards = []
    for row in rows:
        query_id = html.escape(row['query_id'])
        raw = (ROOT / row['image']).read_bytes()
        image = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
        description = html.escape(row['observation'].get('text') or row['observation'].get('warning') or '')
        old = reviews.get(row['query_id'], {})
        controls = []
        for field in REVIEW_FIELDS:
            options = '<option value="">Choose</option>' + ''.join(
                f'<option value="{value}"{" selected" if old.get(field) == value else ""}>{value.title()}</option>'
                for value in ('yes', 'no'))
            controls.append(f'<label>{html.escape(field.replace("_", " ").title())}<select data-field="{field}">{options}</select></label>')
        note = html.escape(old.get('notes', ''), quote=True)
        cards.append(f'<article data-query="{query_id}" data-image="{html.escape(row["image"], quote=True)}" data-description="{html.escape(row["observation"].get("text") or row["observation"].get("warning") or "", quote=True)}"><h2>{query_id}</h2><img src="{image}" alt="Query {query_id}"><p>{description}</p><div class="fields">{"".join(controls)}</div><label>Notes<input data-field="notes" value="{note}"></label></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>ShoeLens visual description review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#193649}h1{margin-bottom:.25rem}p.lead{color:#526b79}article{border:1px solid #ccdbe2;border-radius:12px;padding:1rem;margin:1.2rem 0;background:#f8fbfc}article img{max-width:330px;max-height:280px;object-fit:contain;float:left;margin:0 1.5rem 1rem 0}article:after{content:"";display:block;clear:both}.fields{display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:.7rem}label{display:block;font-weight:600;font-size:.9rem}select,input{display:block;width:100%;box-sizing:border-box;margin-top:.3rem;padding:.5rem;font:inherit}button{background:#267194;color:white;border:0;border-radius:7px;padding:.7rem 1rem;cursor:pointer}</style>
<h1>ShoeLens visual description review</h1><p class="lead">Compare each generated description with its image. Mark Yes only when the field is visually supported. For Unsupported Claims, Yes means the description includes an unsupported inference. Download the CSV and replace artifacts/visual-human-review.csv, then run <code>.venv/bin/python evaluate_visual.py</code> to calculate human-review metrics without API calls.</p>
<button onclick="downloadReview()">Download completed review CSV</button>''' + ''.join(cards) + '''
<script>function quote(s){return '"'+String(s??'').replaceAll('"','""')+'"'}function downloadReview(){const fields=['query_id','image','description','shoe_type_correct','colours_correct','closure_correct','visible_details_correct','unsupported_claims','notes'];const lines=[fields.map(quote).join(',')];for(const a of document.querySelectorAll('article[data-query]')){const values=[a.dataset.query,a.dataset.image,a.dataset.description,...fields.slice(3).map(k=>a.querySelector('[data-field="'+k+'"]').value)];lines.push(values.map(quote).join(','))}const blob=new Blob([lines.join('\\n')+'\\n'],{type:'text/csv;charset=utf-8'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='visual-human-review.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}</script></html>'''
    REVIEW_HTML.write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, default=0,
                        help='Maximum new image API calls; 0 refreshes reports only.')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images must be nonnegative')
    load_env()
    if args.max_images and not os.getenv('OPENROUTER_API_KEY'):
        parser.error('Configure OPENROUTER_API_KEY before paid calls.')

    data = json.loads(PREDICTIONS.read_text())
    report = json.loads((ART / 'evaluation.json').read_text())
    threshold = report['results']['clip']['calibration']
    if data['index_signature'] != threshold['signature']:
        parser.error('Stale predictions; rerun the image benchmark.')
    test = [r for r in data['rows'] if r['method'] == 'clip' and r['variant'] == 'transformed' and r['split'] == 'test']
    uncertain = [r for r in test if r['is_footwear'] and
                 (r['score'] < threshold['threshold'] or r['margin'] < threshold['margin'])]
    model = os.getenv('OPENROUTER_VISION_MODEL') or 'google/gemini-2.5-flash-lite'
    saved = {r['query_id']: r for r in json.loads(RESULTS.read_text())} if RESULTS.exists() else {}
    rows = []
    new_calls = 0
    for row in uncertain:
        image_path = ROOT / row['image']
        raw = image_path.read_bytes()
        image_hash = hashlib.sha256(raw).hexdigest()
        old = saved.get(row['query_id'])
        if old and old.get('image_sha256') == image_hash and old.get('model') == model:
            result = {**old, 'retrieval_seconds': row['seconds'], 'reused': True}
        elif new_calls < args.max_images:
            started = time.perf_counter()
            observation = describe_image(raw)
            result = {'query_id': row['query_id'], 'image': row['image'],
                      'image_sha256': image_hash, 'model': model,
                      'observation': observation,
                      'wall_seconds': round(time.perf_counter() - started, 3),
                      'retrieval_seconds': row['seconds'], 'reused': False}
            saved[row['query_id']] = {k: v for k, v in result.items() if k != 'reused'}
            save_json(RESULTS, list(saved.values()))
            new_calls += 1
            print(row['query_id'], 'generated' if observation.get('text') else 'failed', flush=True)
        else:
            continue
        result['end_to_end_seconds'] = result['retrieval_seconds'] + result['wall_seconds']
        saved[row['query_id']] = {k: v for k, v in result.items() if k != 'reused'}
        rows.append(result)
    if saved:
        save_json(RESULTS, list(saved.values()))

    previous_reviews = {}
    if REVIEWS.exists():
        with REVIEWS.open(newline='', encoding='utf-8-sig') as file:
            previous_reviews = {r['query_id']: r for r in csv.DictReader(file)}
    with REVIEWS.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            old = previous_reviews.get(row['query_id'], {})
            writer.writerow({'query_id': row['query_id'], 'image': row['image'],
                             'description': row['observation'].get('text') or row['observation'].get('warning'),
                             **{key: old.get(key, '') for key in REVIEW_FIELDS},
                             'notes': old.get('notes', '')})
    with REVIEWS.open(newline='', encoding='utf-8') as file:
        reviews = {r['query_id']: r for r in csv.DictReader(file)}
    write_review_html(rows, reviews)
    metrics = visual_metrics(rows, len(uncertain), reviews)
    metrics['model'] = model
    metrics['sample_selection'] = 'All held-out transformed queries classified as footwear but not accepted as exact catalog identities.'
    save_json(REPORT, metrics)
    report['visual_evaluation'] = metrics
    save_json(ART / 'evaluation.json', report)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
