import unittest
import numpy as np
from geometry import support
class GeometricTests(unittest.TestCase):
    def test_consistent_affine_matches(self):
        rng=np.random.default_rng(12)
        points=rng.uniform(20,180,(30,2)).astype('float32')
        descriptors=rng.uniform(0,255,(30,128)).astype('float32')
        angle=.12;m=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],np.float32)
        dest=points@m.T+np.array([4,8],np.float32)
        score,n=support(points,descriptors,dest,descriptors.copy())
        self.assertEqual(n,30);self.assertEqual(score,1)
    def test_too_few_features_returns_zero(self):
        p=np.zeros((3,2),np.float32);d=np.zeros((3,128),np.float32)
        self.assertEqual(support(p,d,p,d),(0.,0))
    def test_unrelated_descriptors_not_evidence(self):
        rng=np.random.default_rng(0);p=rng.uniform(0,200,(30,2)).astype('float32');a=rng.uniform(0,255,(30,128)).astype('float32');b=rng.uniform(0,255,(30,128)).astype('float32')
        self.assertEqual(support(p,a,p,b)[0],0)
if __name__=='__main__':unittest.main()
