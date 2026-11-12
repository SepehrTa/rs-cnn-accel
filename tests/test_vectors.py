"""Export selection and integrity checks, independent of PYNQ/Vivado."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'host'), str(ROOT/'model'), str(ROOT/'scripts')]
from protocol import load_vector
from nn_numpy import forward_int
from run_rtl import network, Vectors


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'model.npz'
        # Distinct images detect accidental first-image selection.
        self.graph = [dict(op='conv', name='stem', src='input', dst='features',
                           k=1, stride=1, pad=0, relu=False),
                      dict(op='gap', name='pool', src='features', dst='pool'),
                      dict(op='fc', name='fc', src='pool', dst='logits')]
        self.q = {'stem': dict(w=np.eye(2, dtype=np.int8).reshape(2,2,1,1),
                              b=np.zeros(2, np.int32), M=np.full(2,2**30), S=np.full(2,-1)),
                  'fc': dict(w=np.array([[1, -2], [-3, 1]], np.int8),
                            b=np.array([7, -9], np.int32))}
        self.x = np.arange(-8, 8, dtype=np.int32).reshape(2, 2, 2, 2)
        self.t = forward_int(self.q, self.graph, self.x)
        self.export = {'graph': np.array(json.dumps(self.graph))}
        self.export.update({f'{n}.{k}': v for n,p in self.q.items() for k,v in p.items()})
        self.export.update({f'vec.{n}': v.copy() for n,v in self.t.items()})
        self.save()

    def save(self):
        np.savez(self.path, **self.export)

    def test_select_second_image(self):
        _, _, got = load_vector(self.path, 1)
        for name, value in self.t.items():
            np.testing.assert_array_equal(got[name], value[1:2])
        self.assertFalse(np.array_equal(got['logits'], self.t['logits'][:1]))

    def test_reject_invalid_indices_and_empty_batch(self):
        for image in (-1, 2, 100):
            with self.assertRaisesRegex(ValueError, 'image index'):
                load_vector(self.path, image)
        self.export['vec.input'] = self.x[:0]
        self.save()
        with self.assertRaisesRegex(ValueError, 'nonempty NCHW'):
            load_vector(self.path)

    def test_reject_corrupt_selected_intermediate(self):
        self.export['vec.pool'][1, 0] += 1
        self.save()
        load_vector(self.path, 0)
        with self.assertRaisesRegex(ValueError, 'vec.pool: image 1 differs'):
            load_vector(self.path, 1)

    def test_reject_missing_and_wrong_shape(self):
        del self.export['vec.logits']
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing vec.logits'):
            load_vector(self.path)
        self.export['vec.logits'] = self.t['logits'][:1]
        self.save()
        with self.assertRaisesRegex(ValueError, 'invalid exported tensor shape'):
            load_vector(self.path)

    def test_reject_input_before_int8_wrap(self):
        self.export['vec.input'][1, 0, 0, 0] = 128
        self.save()
        with self.assertRaisesRegex(ValueError, 'outside signed INT8'):
            load_vector(self.path, 1)
        self.export['vec.input'] = self.x.astype(float)
        self.save()
        with self.assertRaisesRegex(ValueError, 'integer activations'):
            load_vector(self.path, 1)

    def test_rtl_vectors_use_selected_image(self):
        from test_protocol import interpret
        folder = Path(self.temp.name)/'vectors'
        v = Vectors(folder)
        try:
            network(v, self.path, 1)
        finally:
            v.close()
        with np.load(folder/'golden.npz') as z:
            np.testing.assert_array_equal(z['logits'], self.t['logits'][1:2])
        manifest = json.loads((folder/'manifest.json').read_text())
        self.assertEqual(manifest['source']['image'],1)
        self.assertFalse(manifest['source']['synthetic'])
        # Resolve dependent references from actual scalar-interpreter outputs.
        # This catches mismatched image selection in both packets and goldens.
        tokens = [int(s,16) for s in (folder/'stim.hex').read_text().splitlines()]
        expected = [int(s,16) for s in (folder/'expect.hex').read_text().splitlines()]
        offset = 0
        actual = []
        lengths = iter((28, 24, 24))
        for end, value in enumerate(expected):
            if value >> 32:
                count = end+1-len(actual)
                # This fixture's three packet payload lengths: conv, pool, FC.
                length = next(lengths)
                packet = [actual[t & 0xffffffff] if t >> 32 else t for t in tokens[offset:offset+length]]
                got = interpret(np.array(packet,np.uint32),count)
                actual.extend(int(x) & 0xffffffff for x in got)
                offset += length
        self.assertEqual(actual, [x & 0xffffffff for x in expected])
        self.assertEqual(offset, len(tokens))

    def test_board_cli_rejects_index_before_overlay_import(self):
        from accelerator import main
        with patch.object(sys, 'argv', ['accelerator.py', 'unused.bit', str(self.path), '--image', '2']):
            with self.assertRaisesRegex(ValueError, 'image index'):
                main()


if __name__ == '__main__':
    unittest.main()
