"""P0: opt-in main-attention LSE reuse for the existing eager ASI controller."""

from cosmos_framework.inference.asi_existing_ablation import DecoderMetadataController


class MainLSEController(DecoderMetadataController):
    def __enter__(self):
        super().__enter__()
        self._main_lse_previous = []
        for layer in self.layers:
            attn = layer.self_attn
            existed = hasattr(attn, "_asi_reuse_main_lse")
            self._main_lse_previous.append((attn, existed, getattr(attn, "_asi_reuse_main_lse", None)))
            attn._asi_reuse_main_lse = True
        return self

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            for attn, existed, value in self._main_lse_previous:
                if existed:
                    attn._asi_reuse_main_lse = value
                else:
                    delattr(attn, "_asi_reuse_main_lse")
