import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from hash_baseline import HandcraftedBaseline, dhash


def fixture(kind):
    image = Image.new('RGB', (64, 64), 'white')
    draw = ImageDraw.Draw(image)
    if kind == 'vertical':
        draw.rectangle((8, 5, 24, 58), fill='black')
    else:
        draw.rectangle((5, 8, 58, 24), fill='black')
    return image


class HashBaselineTests(unittest.TestCase):
    def test_hash_is_deterministic_and_distinguishes_shapes(self):
        self.assertEqual(dhash(fixture('vertical')), dhash(fixture('vertical')))
        self.assertNotEqual(dhash(fixture('vertical')), dhash(fixture('horizontal')))

    def test_ranking_uses_only_gallery_images(self):
        with tempfile.TemporaryDirectory() as directory:
            gallery = Path(directory)
            fixture('vertical').save(gallery / '1.jpg')
            fixture('horizontal').save(gallery / '2.jpg')
            rows = [{'id': '1'}, {'id': '2'}]
            baseline = HandcraftedBaseline(rows, gallery)
            hits = baseline.rank(fixture('vertical'), k=2)
        self.assertEqual([hit['record']['id'] for hit in hits], ['1', '2'])
        self.assertGreater(hits[0]['score'], 0.99)
        self.assertGreater(hits[0]['score'], hits[1]['score'])

    def test_colour_breaks_a_shape_hash_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            gallery = Path(directory)
            blue = Image.new('RGB', (64, 64), 'white')
            red = Image.new('RGB', (64, 64), 'white')
            ImageDraw.Draw(blue).rectangle((8, 8, 54, 54), fill='blue')
            ImageDraw.Draw(red).rectangle((8, 8, 54, 54), fill='red')
            blue.save(gallery / '1.jpg')
            red.save(gallery / '2.jpg')
            baseline = HandcraftedBaseline([{'id': '1'}, {'id': '2'}], gallery)
            hits = baseline.rank(red, k=2)
        self.assertEqual([hit['record']['id'] for hit in hits], ['2', '1'])


if __name__ == '__main__':
    unittest.main()
