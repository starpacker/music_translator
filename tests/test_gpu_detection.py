"""GPU auto detection must recognize Linux CUDA and Windows DirectML."""
import unittest
from unittest.mock import patch
from core.omr.homr_runner import _homr_gpu_available


class GPUDetectionTest(unittest.TestCase):
    def test_supported_gpu_providers(self):
        for provider in ('CUDAExecutionProvider', 'DmlExecutionProvider'):
            with self.subTest(provider=provider), patch(
                    'onnxruntime.get_available_providers', return_value=[provider, 'CPUExecutionProvider']):
                self.assertTrue(_homr_gpu_available())

    def test_cpu_provider(self):
        with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
            self.assertFalse(_homr_gpu_available())
