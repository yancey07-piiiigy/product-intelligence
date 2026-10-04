import unittest

from gallery_subset import allocate_quotas, stratified_ids


def row(pid, article, gender):
    return {'id': str(pid), 'articleType': article, 'gender': gender}


class GallerySubsetTests(unittest.TestCase):
    def test_quotas_are_proportional_and_exact(self):
        sizes = {('Shoes', 'Men'): 80, ('Heels', 'Women'): 20}
        quotas = allocate_quotas(sizes, {}, 25)
        self.assertEqual(sum(quotas.values()), 25)
        self.assertEqual(quotas[('Shoes', 'Men')], 20)
        self.assertEqual(quotas[('Heels', 'Women')], 5)

    def test_required_evaluation_ids_are_always_kept(self):
        rows = [row(i, 'Shoes', 'Men') for i in range(80)] + [row(i, 'Heels', 'Women') for i in range(80, 100)]
        chosen, _ = stratified_ids(rows, {'0', '1', '99'}, target=20, seed=6201)
        self.assertEqual(len(chosen), 20)
        self.assertTrue({'0', '1', '99'}.issubset(chosen))

    def test_sampling_is_reproducible(self):
        rows = [row(i, 'Shoes' if i % 2 else 'Sandals', 'Men') for i in range(50)]
        a, _ = stratified_ids(rows, {'0'}, target=15, seed=7)
        b, _ = stratified_ids(rows, {'0'}, target=15, seed=7)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
