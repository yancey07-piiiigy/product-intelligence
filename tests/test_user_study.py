"""The human-study analyzer must not reward fast wrong guesses."""
import unittest
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from user_study import analyze_rows, assignment, load_tasks, manual_catalog, public_manual_page, study


TASKS = [
    {'task_id': 'F_A', 'pair_id': 'F', 'query_id': 'q037', 'expected_id': '33702', 'articleType': 'Formal Shoes', 'baseColour': 'Black', 'gender': 'Men', 'season': 'Winter', 'usage': 'Smart Casual'},
    {'task_id': 'F_B', 'pair_id': 'F', 'query_id': 'q042', 'expected_id': '10281', 'articleType': 'Formal Shoes', 'baseColour': 'Black', 'gender': 'Men', 'season': 'Fall', 'usage': 'Formal'},
]


def row(participant, condition, task_id, seconds, selected, season, usage, abstained='0'):
    return {'participant_code': participant, 'pair_id': 'F', 'condition': condition,
            'task_id': task_id, 'seconds': str(seconds), 'selected_product_id': selected,
            'season_answer': season, 'usage_answer': usage, 'abstained': abstained}


class UserStudyTests(unittest.TestCase):
    def test_schedule_balances_product_assignment_across_participants(self):
        first = assignment('P01', TASKS)
        second = assignment('P02', TASKS)
        self.assertEqual({r['condition'] for r in first}, {'manual', 'assistant'})
        self.assertEqual({r['condition'] for r in second}, {'manual', 'assistant'})
        self.assertNotEqual(next(r['task_id'] for r in first if r['condition'] == 'manual'),
                            next(r['task_id'] for r in second if r['condition'] == 'manual'))

    def test_full_schedule_swaps_each_pair_between_first_two_participants(self):
        tasks = load_tasks()
        first = {r['pair_id']: r['task_id'] for r in assignment('P01', tasks) if r['condition'] == 'manual'}
        second = {r['pair_id']: r['task_id'] for r in assignment('P02', tasks) if r['condition'] == 'manual'}
        self.assertEqual(set(first), set(second))
        self.assertTrue(all(first[pair] != second[pair] for pair in first))

    def test_manual_catalog_has_only_same_category_colour_and_audience(self):
        rows = [
            {'id': '33702', 'articleType': 'Formal Shoes', 'baseColour': 'Black', 'gender': 'Men'},
            {'id': 'other', 'articleType': 'Formal Shoes', 'baseColour': 'Black', 'gender': 'Men'},
            {'id': 'wrong', 'articleType': 'Sports Shoes', 'baseColour': 'Black', 'gender': 'Men'},
        ]
        self.assertEqual({r['id'] for r in manual_catalog(TASKS[0], rows)}, {'33702', 'other'})

    def test_manual_catalog_uses_frozen_twenty_item_pool_when_present(self):
        task = {**TASKS[0], 'manual_candidate_ids': ['33702', 'near']}
        rows = [
            {'id': value, 'articleType': 'Formal Shoes', 'baseColour': 'Black', 'gender': 'Men'}
            for value in ('33702', 'near', 'not_in_pool')
        ]
        self.assertEqual({r['id'] for r in manual_catalog(task, rows)}, {'33702', 'near'})

    def test_manual_page_exposes_filter_but_not_answer_key(self):
        rows = [{'id': '33702', 'articleType': 'Formal Shoes', 'baseColour': 'Black',
                 'gender': 'Men', 'season': 'Winter', 'usage': 'Smart Casual'}]
        page = public_manual_page(TASKS[0], rows, 1)
        self.assertEqual(page['task']['query_id'], 'q037')
        self.assertNotIn('expected_id', page['task'])
        self.assertEqual(page['total'], 1)

    def test_manual_api_returns_public_task_only(self):
        import server
        task = next(t for t in load_tasks() if t['task_id'] == 'F_A')
        samples = [{'id': id_, 'articleType': task['articleType'],
                    'baseColour': task['baseColour'], 'gender': task['gender']}
                   for id_ in task['manual_candidate_ids']]
        handler = server.Handler.__new__(server.Handler)
        handler.path = '/api/study/manual?task=F_A&page=1'
        replies = []
        handler.json = lambda value, status=200: replies.append((status, value))
        with patch.object(server, 'search', SimpleNamespace(rows=samples)):
            handler.do_GET()
        self.assertEqual(replies[0][0], 200)
        self.assertNotIn('expected_id', replies[0][1]['task'])
        self.assertEqual(replies[0][1]['task']['query_id'], 'q037')

    def test_wrong_fast_guess_gets_time_cap_and_cannot_meet_target(self):
        # P01's expected assignment is assistant A, manual B.
        rows = [row('P01', 'assistant', 'F_A', 2, 'wrong', 'Winter', 'Smart Casual'),
                row('P01', 'manual', 'F_B', 100, '10281', 'Fall', 'Formal')]
        result = analyze_rows(rows, TASKS, time_limit=180)
        self.assertEqual(result['assistant_success_rate'], 0)
        self.assertEqual(result['manual_success_rate'], 1)
        self.assertEqual(result['assistant_capped_seconds'], [180])
        self.assertFalse(result['target_met'])

    def test_identity_and_supported_fields_are_both_required(self):
        rows = [row('P01', 'assistant', 'F_A', 40, '33702', 'Summer', 'Smart Casual'),
                row('P01', 'manual', 'F_B', 100, '10281', 'Fall', 'Formal')]
        result = analyze_rows(rows, TASKS)
        self.assertEqual(result['assistant_success_rate'], 0)

    def test_time_target_requires_both_conditions_to_finish_correctly(self):
        rows = [row('P01', 'assistant', 'F_A', 40, '33702', 'Winter', 'Smart Casual'),
                row('P01', 'manual', 'F_B', 100, 'wrong', 'Winter', 'Formal')]
        result = analyze_rows(rows, TASKS, min_participants=1, min_both_correct_pairs=1)
        self.assertFalse(result['target_met'])

    def test_time_target_can_pass_with_faster_correct_assistant(self):
        rows = [row('P01', 'assistant', 'F_A', 40, '33702', 'Winter', 'Smart Casual'),
                row('P01', 'manual', 'F_B', 100, '10281', 'Fall', 'Formal')]
        result = analyze_rows(rows, TASKS, min_participants=1, min_both_correct_pairs=1)
        self.assertAlmostEqual(result['median_both_correct_time_reduction'], .6)
        self.assertTrue(result['target_met'])

    def test_incomplete_pairs_are_reported_not_scored(self):
        rows = [row('P01', 'assistant', 'F_A', 40, '33702', 'Winter', 'Smart Casual')]
        result = analyze_rows(rows, TASKS)
        self.assertEqual(result['complete_pairs'], 0)
        self.assertEqual(result['incomplete_pairs'], 1)
        self.assertIsNone(result['median_paired_time_reduction'])

    def test_missing_condition_is_derived_from_frozen_assignment(self):
        rows = [row('P01', 'assistant', 'F_A', 40, '33702', 'Winter', 'Smart Casual'),
                row('P01', 'manual', 'F_B', 100, '10281', 'Fall', 'Formal')]
        for item in rows:
            del item['condition']
        result = analyze_rows(rows, TASKS, min_participants=1, min_both_correct_pairs=1)
        self.assertEqual(result['complete_pairs'], 1)
        self.assertEqual(result['manual_success_rate'], 1)

    def test_explicit_wrong_condition_remains_rejected(self):
        wrong = row('P01', 'manual', 'F_A', 40, '33702', 'Winter', 'Smart Casual')
        with self.assertRaises(ValueError):
            analyze_rows([wrong], TASKS)

    def test_reanalysis_uses_hash_matched_provenance_instead_of_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / 'observations.csv'
            with csv_path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=row('P01', 'assistant', 'F_A', 40, '33702', 'Winter', 'Smart Casual').keys())
                writer.writeheader()
                writer.writerow(row('P01', 'assistant', 'F_A', 40, '33702', 'Winter', 'Smart Casual'))
                writer.writerow(row('P01', 'manual', 'F_B', 100, '10281', 'Fall', 'Formal'))
            tasks_path = root / 'tasks.json'
            tasks_path.write_text(json.dumps({'tasks': TASKS}))
            output_path = root / 'results.json'
            output_path.write_text(json.dumps({'input_file': str(csv_path), 'data_provenance': 'stale incorrect claim', 'reporting_status': 'unverified'}))
            provenance_path = root / 'provenance.json'
            provenance_path.write_text(json.dumps({
                'input_csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                'data_provenance': 'Participant observations verified by project owner.',
                'reporting_status': 'project_owner_verified_participant_observations',
            }))
            result = study(csv_path, tasks_path=tasks_path, output_path=output_path, provenance_path=provenance_path)
            self.assertEqual(result['data_provenance'], 'Participant observations verified by project owner.')
            self.assertEqual(result['reporting_status'], 'project_owner_verified_participant_observations')
            self.assertEqual(json.loads(output_path.read_text())['reporting_status'], result['reporting_status'])
            csv_path.write_bytes(csv_path.read_bytes() + b'\n')
            changed = study(csv_path, tasks_path=tasks_path, output_path=output_path, provenance_path=provenance_path)
            self.assertEqual(changed['reporting_status'], 'provenance_unconfirmed')


if __name__ == '__main__': unittest.main()
