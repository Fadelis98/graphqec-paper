import os
import unittest

os.environ.setdefault('QEC_DATA_PATH', '/tmp/qec_dummy_data')

from graphqec.qecc import ETHBBCode


class TestBBNoiseModels(unittest.TestCase):
    def setUp(self):
        self.code = ETHBBCode.from_profile(
            '[[72,12,6]]', logical_basis='Z', check_basis='ZX'
        )

    def test_explicit_uniform_is_byte_identical_to_published_path(self):
        legacy = self.code.get_syndrome_circuit(5, physical_error_rate=0.001)
        explicit = self.code.get_syndrome_circuit(
            5, physical_error_rate=0.001,
            noise_model='uniform_depolarizing',
        )
        self.assertEqual(str(legacy), str(explicit))

    def test_si1000_preserves_detector_and_observable_layout(self):
        uniform = self.code.get_syndrome_circuit(5, physical_error_rate=0.001)
        si1000 = self.code.get_syndrome_circuit(
            5, physical_error_rate=0.001, noise_model='si1000'
        )
        self.assertEqual(uniform.num_detectors, si1000.num_detectors)
        self.assertEqual(uniform.num_observables, si1000.num_observables)
        self.assertGreater(si1000.detector_error_model().num_errors, 0)

    def test_constructor_selects_si1000_for_evaluation_pipeline(self):
        code = ETHBBCode.from_profile(
            '[[72,12,6]]', check_basis='ZX', noise_model='si1000'
        )
        circuit = code.get_syndrome_circuit(5, physical_error_rate=0.001)
        self.assertEqual(code.noise_model, 'si1000')
        self.assertIn('MR(0.005)', str(circuit))


if __name__ == '__main__':
    unittest.main()
