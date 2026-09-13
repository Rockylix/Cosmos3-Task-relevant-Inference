"""Check CLI-to-controller wiring without loading a model or using CUDA."""

from types import SimpleNamespace

import pytest

from cosmos_framework.scripts import action_policy_server_robolab_version1 as server


@pytest.mark.parametrize(
    "mode,compiled,graph",
    [
        ("legacy", False, False),
        ("optimized-eager", False, False),
        ("compile", True, False),
        ("compile-graph", True, True),
    ],
)
def test_setup_and_request_controller(monkeypatch, mode, compiled, graph):
    monkeypatch.setattr(
        server.RobolabPolicyService,
        "_build_setup_args",
        lambda self, args: SimpleNamespace(model_copy=lambda update: update),
    )
    args = server.Version1ServerArgs(asi_execution=mode, num_steps=4, guidance=3, shift=5)
    service = object.__new__(server.Version1PolicyService)
    assert service._build_setup_args(args) == {"use_torch_compile": compiled, "use_cuda_graphs": graph}
    controllers, requests = [], []

    class Controller:
        def __init__(self, **kwargs):
            controllers.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def finish(self):
            return {"strategy_version": "test"}

    monkeypatch.setattr(server, "Version1Controller", Controller)
    monkeypatch.setattr(server, "OptimizedVersion1Controller", Controller)

    def init(self, args):
        self.cfg = SimpleNamespace(guidance=3, num_steps=4)
        self.model = SimpleNamespace(net="model", generate_samples_from_batch=lambda *a, **kw: requests.append((a, kw)))

    monkeypatch.setattr(server.RobolabPolicyService, "__init__", init)
    service = server.Version1PolicyService(args)
    for seed in [1, 2]:
        service.model.generate_samples_from_batch("input", guidance=3, num_steps=4, seed=[seed])
    assert service._request_count == 2
    assert len(controllers) == 2  # fresh request-local controller, never stale masks
    assert [kw["seed"] for _, kw in requests] == [[1], [2]]
    for cfg in controllers:
        if mode == "legacy":
            assert "optimized" not in cfg
        else:
            assert cfg["optimized"] and cfg["cache_layout"] and cfg["compile_profile_decoder"]
            assert cfg["cuda_graphs"] == graph
            assert cfg["compile_profile_kernel"] is False


def test_compiled_capture_rejected():
    with pytest.raises(ValueError, match="Eager attention/block capture"):
        server.Version1ServerArgs(
            asi_execution="compile",
            hidden_state_capture_dir="/tmp/test-capture",
            hidden_state_capture_chunks=[3],
            num_steps=4,
        )
