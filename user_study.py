"""Analyze an image-first, paired product-disambiguation study.

The same target photo is shown without its catalog ID. Manual participants browse
a fixed same-category/colour/audience gallery; assistant participants use ShoeLens.
Only anonymized task outcomes are analyzed; the CSV's provenance is reported separately.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from core import ROOT

TASKS_PATH = ROOT / 'data/user-study-tasks.json'
PROVENANCE_PATH = ROOT / 'artifacts/user-study-provenance.json'
FIELDS = ('participant_code', 'pair_id', 'condition', 'task_id', 'seconds',
          'selected_product_id', 'season_answer', 'usage_answer', 'abstained')
REQUIRED_FIELDS = tuple(field for field in FIELDS if field != 'condition')
TIME_LIMIT = 180


def load_tasks(path=TASKS_PATH):
    tasks = json.loads(Path(path).read_text())['tasks']
    by_pair = defaultdict(list)
    for task in tasks:
        by_pair[task['pair_id']].append(task)
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate study task ID')
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or len({t['expected_id'] for t in pair}) != 2:
            raise ValueError(f'Pair {pair_id} needs two different products')
        if len({(t['articleType'], t['baseColour'], t['gender']) for t in pair}) != 1:
            raise ValueError(f'Pair {pair_id} must share category, colour and audience')
    return tasks


def manual_catalog(task, catalog_rows):
    """A stable, non-name-sorted pool; the answer ID is not used for ordering."""
    allowed = set(task['manual_candidate_ids']) if 'manual_candidate_ids' in task else None
    pool = [r for r in catalog_rows if all(r[k] == task[k] for k in
            ('articleType', 'baseColour', 'gender')) and
            (allowed is None or r['id'] in allowed)]
    if allowed is not None and ({r['id'] for r in pool} != allowed or task['expected_id'] not in allowed):
        raise ValueError(f"Study pool missing a frozen candidate for {task['task_id']}")
    return sorted(pool, key=lambda r: hashlib.sha256(
        (task['task_id'] + ':' + r['id']).encode()).hexdigest())


def public_manual_page(task, catalog_rows, page, page_size=24):
    """Return the fixed manual browse task without exposing its answer key."""
    pool = manual_catalog(task, catalog_rows)
    page = max(1, int(page))
    start = (page - 1) * page_size
    return {
        'task': {key: task[key] for key in
                 ('task_id', 'pair_id', 'query_id', 'articleType', 'baseColour', 'gender')},
        'total': len(pool), 'page': page, 'page_size': page_size,
        'items': pool[start:start + page_size],
    }


def assignment(participant_code, tasks):
    match = re.fullmatch(r'P(\d+)', participant_code)
    if not match or int(match.group(1)) < 1:
        raise ValueError('Use an anonymous participant code such as P01')
    number = int(match.group(1))
    pairs = defaultdict(list)
    for task in tasks:
        pairs[task['pair_id']].append(task)
    original_order = sorted(pairs)
    order = original_order[:]
    shift = (number - 1) % len(order)
    order = order[shift:] + order[:shift]
    trials = []
    for i, pair_id in enumerate(order):
        a, b = sorted(pairs[pair_id], key=lambda t: t['task_id'])
        manual, assistant = (b, a) if (number + original_order.index(pair_id)) % 2 else (a, b)
        pair_trials = [{'pair_id': pair_id, 'condition': 'manual', 'task_id': manual['task_id']},
                       {'pair_id': pair_id, 'condition': 'assistant', 'task_id': assistant['task_id']}]
        if (number + i) % 2:
            pair_trials.reverse()
        trials.extend(pair_trials)
    by_id = {task['task_id']: task for task in tasks}
    for trial in trials:
        task = by_id[trial['task_id']]
        trial['image'] = task.get('image', f"artifacts/queries/{task['query_id']}.jpg")
        trial['clues'] = {key: task[key] for key in ('articleType', 'baseColour', 'gender')}
        if trial['condition'] == 'manual':
            trial['manual_url'] = 'http://127.0.0.1:8501/?study=manual&task=' + task['task_id']
    return trials


def _norm(value):
    return ' '.join((value or '').strip().casefold().split())


def analyze_rows(rows, tasks, time_limit=TIME_LIMIT, min_participants=5,
                 min_both_correct_pairs=8):
    lookup = {t['task_id']: t for t in tasks}
    observed = defaultdict(dict)
    participants = set()
    assignments = {}
    field_errors = defaultdict(Counter)
    raw_times = defaultdict(list)
    for row in rows:
        if not any((value or '').strip() for value in row.values()):
            continue
        code, pair_id, task_id = (row[k].strip() for k in
                ('participant_code', 'pair_id', 'task_id'))
        if task_id not in lookup or lookup[task_id]['pair_id'] != pair_id:
            raise ValueError(f'Unknown or mismatched task: {task_id}')
        if code not in assignments:
            assignments[code] = assignment(code, tasks)
        scheduled = {(t['pair_id'], t['condition'], t['task_id'])
                     for t in assignments[code]}
        condition = (row.get('condition') or '').strip()
        if not condition:
            condition = next((t['condition'] for t in assignments[code]
                              if t['pair_id'] == pair_id and t['task_id'] == task_id), '')
        if (pair_id, condition, task_id) not in scheduled:
            raise ValueError(f'{code}: task {task_id} is not assigned to {condition}')
        if condition in observed[(code, pair_id)]:
            raise ValueError(f'Duplicate {condition} result for {code}/{pair_id}')
        try:
            seconds = float(row['seconds'])
        except (TypeError, ValueError):
            raise ValueError(f'{code}/{task_id}: seconds must be recorded') from None
        if not 0 < seconds <= time_limit:
            raise ValueError(f'{code}/{task_id}: record 0 < seconds <= {time_limit}')
        if row['abstained'] not in ('0', '1'):
            raise ValueError(f'{code}/{task_id}: abstained must be 0 or 1')
        task = lookup[task_id]
        checks = {'product': _norm(row['selected_product_id']) == _norm(task['expected_id']),
                  'season': _norm(row['season_answer']) == _norm(task['season']),
                  'usage': _norm(row['usage_answer']) == _norm(task['usage'])}
        success = row['abstained'] == '0' and all(checks.values())
        for field, correct in checks.items():
            if not correct:
                field_errors[condition][field] += 1
        raw_times[condition].append(seconds)
        observed[(code, pair_id)][condition] = {
            'seconds': seconds, 'capped_seconds': seconds if success else time_limit,
            'success': success, 'abstained': row['abstained'] == '1',
            'timeout': seconds == time_limit and not success,
        }
        participants.add(code)
    complete = [pair for pair in observed.values() if set(pair) == {'manual', 'assistant'}]
    complete_participants = {code for code, pair in observed if
                             all(set(observed.get((code, pid), {})) == {'manual', 'assistant'}
                                 for pid in {t['pair_id'] for t in tasks})}
    reductions = [1 - p['assistant']['capped_seconds'] / p['manual']['capped_seconds']
                  for p in complete]
    both_correct = [1 - p['assistant']['seconds'] / p['manual']['seconds']
                    for p in complete if p['manual']['success'] and p['assistant']['success']]
    manual = [p['manual'] for p in complete]
    assistant = [p['assistant'] for p in complete]
    median = statistics.median(reductions) if reductions else None
    both_correct_median = statistics.median(both_correct) if both_correct else None
    manual_success = sum(p['success'] for p in manual) / len(manual) if manual else None
    assistant_success = sum(p['success'] for p in assistant) / len(assistant) if assistant else None
    return {
        'protocol': 'similar-shoe-image-first-v1', 'time_limit_seconds': time_limit,
        'participants_recorded': len(participants), 'participants_with_all_pairs': len(complete_participants),
        'complete_pairs': len(complete), 'incomplete_pairs': len(observed) - len(complete),
        'median_paired_time_reduction': median,
        'median_both_correct_time_reduction': both_correct_median,
        'both_correct_pairs': len(both_correct),
        'manual_success_rate': manual_success, 'assistant_success_rate': assistant_success,
        'raw_time_median_seconds': {key: statistics.median(raw_times[key]) for key in ('manual', 'assistant') if raw_times[key]},
        'field_error_counts': {key: {field: field_errors[key][field] for field in ('product', 'season', 'usage')}
                               for key in ('manual', 'assistant')},
        'manual_capped_seconds': [p['capped_seconds'] for p in manual],
        'assistant_capped_seconds': [p['capped_seconds'] for p in assistant],
        'manual_abstentions': sum(p['abstained'] for p in manual),
        'assistant_abstentions': sum(p['abstained'] for p in assistant),
        'manual_timeouts': sum(p['timeout'] for p in manual),
        'assistant_timeouts': sum(p['timeout'] for p in assistant),
        'target_time_reduction': 0.5,
        'target_met': (len(complete_participants) >= min_participants and
                       len(complete) == len(observed) and median is not None and median >= 0.5 and
                       len(both_correct) >= min_both_correct_pairs and
                       both_correct_median is not None and both_correct_median >= 0.5 and
                       assistant_success is not None and assistant_success >= manual_success),
        'claim_limit': 'Small, same-source catalog study; not independent phone-photo validation.',
    }


def study(path, tasks_path=TASKS_PATH, output_path=ROOT / 'artifacts/user-study-results.json', provenance_path=PROVENANCE_PATH):
    tasks = load_tasks(tasks_path)
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not set(REQUIRED_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Missing study columns: ' + ', '.join(set(REQUIRED_FIELDS) - set(reader.fieldnames or [])))
        rows = list(reader)
    result = analyze_rows(rows, tasks)
    if not result['complete_pairs']:
        raise ValueError('No complete manual/assistant pairs. Enter real participant observations.')
    result['input_file'] = str(path)
    result['input_csv_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result['condition_inferred_rows'] = sum(not (row.get('condition') or '').strip() for row in rows)
    provenance_file = Path(provenance_path)
    provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
    if provenance.get('input_csv_sha256') == result['input_csv_sha256']:
        result['data_provenance'] = provenance['data_provenance']
        result['reporting_status'] = provenance['reporting_status']
    else:
        result['data_provenance'] = 'No provenance attestation matches this CSV hash.'
        result['reporting_status'] = 'provenance_unconfirmed'
    Path(output_path).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the paired, image-first ShoeLens user study')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--schedule', metavar='P01', help='print the counterbalanced task order')
    group.add_argument('--analyze', type=Path, help='analyze completed CSV rows')
    args = parser.parse_args()
    if args.schedule:
        print(json.dumps(assignment(args.schedule, load_tasks()), indent=2))
    else:
        study(args.analyze)
