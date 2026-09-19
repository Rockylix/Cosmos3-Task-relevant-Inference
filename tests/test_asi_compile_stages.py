import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cosmos_framework.inference.asi_main_lse import MainLSEController
from tools.benchmark_asi_compile_stages import StagedController


class StagedCompileTest(unittest.TestCase):
    def test_sparse_only_compile_preserves_metadata_lse_dense_route(self):
        ctrl = StagedController.__new__(StagedController)
        ctrl.dense_eager = True
        eager = object()
        wrapped = SimpleNamespace(_orig_mod=eager)
        with patch.object(MainLSEController, "_run_dense_profile_layer", return_value="ok") as call:
            self.assertEqual(ctrl._run_dense_profile_layer(decoder_layer=wrapped, block=0), "ok")
        self.assertIs(call.call_args.kwargs["decoder_layer"], eager)
        self.assertEqual(call.call_args.kwargs["block"], 0)
        self.assertNotIn("compile_profile_decoder", ctrl.__dict__)

    def test_full_compile_keeps_compiled_dense_layer(self):
        ctrl = StagedController.__new__(StagedController)
        ctrl.dense_eager = False
        wrapped = SimpleNamespace(_orig_mod=object())
        with patch.object(MainLSEController, "_run_dense_profile_layer", return_value="ok") as call:
            ctrl._run_dense_profile_layer(decoder_layer=wrapped, block=3)
        self.assertIs(call.call_args.kwargs["decoder_layer"], wrapped)


if __name__ == "__main__":
    unittest.main()
