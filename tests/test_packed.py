from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'model')]
from packed import pack_packet,unpack_packet,unpack_output,output_words
from protocol import add_packet,pool_packet,fc_packet,conv_packets


class PackedTests(unittest.TestCase):
    def check_packet(self,packet):
        packed=pack_packet(packet)
        np.testing.assert_array_equal(unpack_packet(packed),packet)
        self.assertEqual(int(packed[15]),1)
        return packed

    def test_exact_signed_bytes_and_tail(self):
        p=pool_packet(np.array([-128,-1,0,127,-3],np.int32).reshape(1,1,5))
        got=self.check_packet(p)
        np.testing.assert_array_equal(got[16:],[0x7f00ff80,0x000000fd])

    def test_residual_interleaving(self):
        p=add_packet(np.array([-128,127,-1]),np.array([0,1,-3]),dict(Ma=1,Mb=1,S=0),False)
        got=self.check_packet(p)
        np.testing.assert_array_equal(got[16:],[0x017f0080,0x0000fdff])

    def test_classifier_keeps_full_int32_and_separate_weight_tails(self):
        p=fc_packet(np.array([-2000000,70000,12345]),dict(w=np.array([[-1,2,-3],[7,-8,9]],np.int8),b=np.array([-2**31+60000000,2**31-60000000])))
        got=self.check_packet(p)
        np.testing.assert_array_equal(got[16:19],p[16:19])
        self.assertEqual(int(got[20]),0x00fd02ff)
        self.assertEqual(int(got[22]),0x0009f807)
        self.assertEqual(len(got),23)

    def test_rectangular_convolution_tiles(self):
        rng=np.random.default_rng(71)
        x=rng.integers(-128,128,(3,5,37),dtype=np.int8)
        q=dict(w=rng.integers(-128,128,(3,3,3,3),dtype=np.int8),b=np.zeros(3,np.int32),M=np.full(3,2**30),S=np.zeros(3,np.int32))
        for packet,_,_ in conv_packets(x,q,dict(k=3,stride=2,pad=1,relu=False),cols=4):
            self.check_packet(packet)

    def test_signed_output_and_int32_results(self):
        for op in (0,1):
            h=np.zeros(16,np.uint32);h[0]=op
            y=unpack_output(h,np.array([0x7f00ff80,0x000000fd],np.uint32),5)
            np.testing.assert_array_equal(y,[-128,-1,0,127,-3])
            self.assertEqual(output_words(h,5),2)
        for op in (2,3):
            h=np.zeros(16,np.uint32);h[0]=op
            np.testing.assert_array_equal(unpack_output(h,np.array([0x80000000,0x7fffffff],np.uint32),2),[-2**31,2**31-1])


if __name__=='__main__':unittest.main()
