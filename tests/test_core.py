import tempfile
import unittest
from pathlib import Path

from calibration import select_threshold
from core import load_catalog


class CatalogAndCalibrationTests(unittest.TestCase):
    def test_calibration_never_claims_success_for_all_wrong(self):
        rows = [{'score': .9, 'margin': .1, 'correct': False, 'is_footwear': True} for _ in range(40)]
        self.assertGreater(select_threshold(rows)['threshold'], 1)

    def test_threshold_can_select_reliable_subset(self):
        rows = ([{'score': .9, 'margin': .1, 'correct': True, 'is_footwear': True} for _ in range(12)] +
                [{'score': .3, 'margin': .01, 'correct': False, 'is_footwear': True} for _ in range(12)])
        self.assertEqual(select_threshold(rows)['accepted'], 12)

    def test_malformed_row_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            header = 'id,masterCategory,articleType,baseColour,gender,season,usage,productDisplayName\n'
            valid = ''.join(f'{i},Footwear,Sports Shoes,Blue,Unisex,Summer,Sports,Shoe {i}\n' for i in range(5))
            path.write_text(header + valid + 'bad,row\n')
            rows, bad = load_catalog(path)
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(bad), 1)

    def test_unquoted_commas_in_name_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName\nX1,Unisex,Footwear,Shoes,Sandals,Green,Summer,2012,Casual,Name, with, commas\n')
            rows, bad = load_catalog(path)
            self.assertEqual(rows[-1]['productDisplayName'], 'Name, with, commas')
            self.assertFalse(bad)


if __name__ == '__main__':
    unittest.main()
