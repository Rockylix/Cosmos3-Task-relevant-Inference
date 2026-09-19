"""Cumulative, eager-only ablations of existing ASI system optimizations.

B2 adds contiguous-layout batched scoring and request-local geometry/layout
reuse to B1, retaining per-layer finite checks. B3 defers those checks to the
unchanged planner. B4 adds the production sparse SequencePack metadata cache.
B5 returns profiles through decoder metadata, as production optimized-eager
does. None of these controllers compiles a module or captures a CUDA graph.

Layout cache hits require the very same packed-sequence object; holding a
reference prevents Python id reuse. The cache lives only for this controller's
request, not across chunks. This assumes that object's token layout is immutable
within a request, as in the existing production adapter (timestep values may
change). Geometry is validated once, not reconstructed in every block.
"""

from cosmos_framework.inference import edge_core_stable as optimized
from cosmos_framework.inference import edge_core_stable_fast as fast
from cosmos_framework.scripts import robolab_version1 as legacy


class GeometryCachedController(legacy.Version1Controller):
    """B2: cached geometry and existing fast scorer, with legacy sparse packing."""

    validate_each_profile = True

    def __init__(self, *, torch, net, guidance, num_steps, output_dir=None):
        super().__init__(torch=torch, net=net, guidance=guidance, num_steps=num_steps, output_dir=output_dir)
        # Only the production pre-hook is reused: the legacy run_layer,
        # SequencePack construction, restoration and planner remain unchanged.
        self.optimized = True
        self.cache_layout = True
        self._layout_cache = {}
        self._profile_geometry = None

    _network_pre_hook = optimized.Version1Controller._network_pre_hook

    def _capture_profile(self, **kwargs):
        block = int(kwargs["layer_index"])
        if self._profile_callback_block != block or self._layout is None:
            raise RuntimeError("Version1 attention callback has stale block state")
        profile = fast.action_aligned_future_profiles(
            kwargs["q_gen"],
            kwargs["k_ar"],
            kwargs["k_gen"],
            kwargs["v_ar"],
            kwargs["v_gen"],
            float(kwargs["scaling"]),
            self._profile_geometry,
        )
        if self.validate_each_profile:
            if not bool(self.torch.isfinite(profile).all()) or bool((profile < 0).any()):
                raise RuntimeError("Version1 fused Action-Relevance profile is invalid")
        self._profile_records.append({"block": block, "profiles": profile.clone()})


class DeferredValidationController(GeometryCachedController):
    """B3: preserve planner's finite gate, remove redundant per-layer host reads."""

    validate_each_profile = False


class MetadataCachedController(optimized.Version1Controller):
    """B4: production sparse metadata cache, still callback-based scoring."""

    profile_via_metadata = False

    def __init__(self, *, torch, net, guidance, num_steps, output_dir=None):
        super().__init__(
            torch=torch,
            net=net,
            guidance=guidance,
            num_steps=num_steps,
            output_dir=output_dir,
            optimized=True,
            compile_profile_decoder=self.profile_via_metadata,
            cuda_graphs=False,
            compile_profile_kernel=False,
            cache_layout=True,
        )


class DecoderMetadataController(MetadataCachedController):
    """B5: production optimized-eager; the decoder itself is NOT compiled.

    ``compile_profile_decoder`` is the existing API's misleading flag name:
    it selects metadata-return plumbing and does not call torch.compile.
    The benchmark separately rejects pre-compiled decoder layers.
    """

    profile_via_metadata = True


CONTROLLERS = {
    "b2": GeometryCachedController,
    "b3": DeferredValidationController,
    "b4": MetadataCachedController,
    "b5": DecoderMetadataController,
}
