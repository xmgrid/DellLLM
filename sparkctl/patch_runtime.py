#!/usr/bin/env python3
"""把解包后的 vLLM 改成能加载 0731 的 DSpark 草稿和视觉权重。

在两台机器上各跑一次。已经打过的位置会跳过，不会重复改写。
视觉前向在 vision_v4.py，这里只负责把它装进模型目录并接上入口。
"""

from pathlib import Path
import shutil
import sys

ROOT = Path("/opt/dspark/usr/local/lib/python3.12/dist-packages/vllm")
MODEL = ROOT / "models/deepseek_v4/nvidia/model.py"
SPEC = ROOT / "config/speculative.py"
HERE = Path(__file__).resolve().parent


def must_replace(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text and old not in text:
        print(f"already patched: {label}")
        return
    if old not in text:
        raise SystemExit(f"anchor missing for {label} in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {label}")


def main() -> None:
    dest = MODEL.parent / "vision_v4.py"
    shutil.copyfile(HERE / "vision_v4.py", dest)
    print(f"installed {dest}")

    must_replace(
        SPEC,
        '''                    self.draft_model_config.hf_config.architectures = [
                        "DSparkDraftModel"
                    ]
                    self.update_arch_()
''',
        '''                    self.draft_model_config.hf_config.architectures = [
                        "DSparkDraftModel"
                    ]
                    # 0731 的草稿是 DSpark（权重名 mtp.*，配置里有 dspark_*）。
                    # update_arch_() 会把 deepseek_v4 改写成 MTP，再因
                    # 推测长度不能被 n_predict 整除而拒绝启动。这里停在
                    # DSparkDraftModel，不调用那次改写。
''',
        "keep DSpark draft architecture",
    )

    must_replace(
        MODEL,
        "                else:\n"
        "                    if is_pp_missing_parameter(name, self):\n"
        "                        continue\n"
        "                    param = params_dict[name]\n",
        "                else:\n"
        "                    if is_pp_missing_parameter(name, self):\n"
        "                        continue\n"
        "                    # 前三层是 hash MoE，没有 gate bias。缺这些张量时跳过，
                    # 不要因此中断加载。aligner / vision 不能放进跳过列表。\n"
        "                    if name not in params_dict and \".ffn.gate.\" in name:\n"
        "                        continue\n"
        "                    param = params_dict[name]\n",
        "hash-layer gate skip",
    )

    text = MODEL.read_text()
    if "SupportsMultiModal" not in text.split("class DeepseekV4ForCausalLM")[0]:
        # Import next to SupportsPP.
        if "SupportsPP," not in text and "SupportsPP" not in text:
            raise SystemExit("SupportsPP import not found")
        text = text.replace(
            "SupportsPP,",
            "SupportsPP, SupportsMultiModal,",
            1,
        )
        # The import line may be `SupportsPP` without a trailing comma on its own.
        if "SupportsMultiModal" not in text:
            text = text.replace("SupportsPP\n", "SupportsPP, SupportsMultiModal\n", 1)
    old_cls = (
        "class DeepseekV4ForCausalLM(\n"
        "    nn.Module, SupportsPP, SupportsEagle3, DeepseekV4MixtureOfExperts\n"
        "):"
    )
    new_cls = (
        "class DeepseekV4ForCausalLM(\n"
        "    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3, DeepseekV4MixtureOfExperts\n"
        "):"
    )
    if old_cls not in text and new_cls not in text:
        raise SystemExit("class header not found")
    text = text.replace(old_cls, new_cls, 1)

    old_init_tail = "        self.set_moe_parameters()\n"
    new_init_tail = '''        self.set_moe_parameters()
        self._init_v4_vision(vllm_config)

'''
    if "self._init_v4_vision" not in text:
        if old_init_tail not in text:
            raise SystemExit("set_moe_parameters anchor missing")
        text = text.replace(old_init_tail, new_init_tail, 1)

    old_embed = '''    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)
'''
    new_embed = '''    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<｜image｜>"
        raise ValueError(f"Unsupported modality: {modality}")

    def embed_multimodal(self, **kwargs: object):
        from .vision_v4 import encode_images

        return encode_images(self, kwargs.get("pixel_values"), kwargs.get("vision_grid"))

    def _init_v4_vision(self, vllm_config: VllmConfig) -> None:
        from .vision_v4 import Aligner, VisionTower

        config = self.config
        hidden = config.hidden_size
        with self._mark_tower_model(vllm_config, "image"):
            self.vision = VisionTower(config)
            self.aligner = Aligner(
                int(getattr(config, "vision_dim", 1024)), hidden
            )
            self.image_start = nn.Parameter(torch.zeros(hidden))
            self.image_end = nn.Parameter(torch.zeros(hidden))
            self.image_newline = nn.Parameter(torch.zeros(hidden))
            self.image_pad = nn.Parameter(torch.zeros(hidden))
        try:
            from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

            tok = cached_tokenizer_from_config(vllm_config.model_config)
            image_id = tok.convert_tokens_to_ids("<｜image｜>")
        except Exception:
            image_id = 129279
        self.image_token_id = int(image_id)
        self.configure_mm_token_handling(config.vocab_size, [self.image_token_id])
'''
    if "def embed_multimodal" not in text:
        if old_embed not in text:
            raise SystemExit("embed_input_ids anchor missing")
        text = text.replace(old_embed, new_embed, 1)

    if "register_v4_multimodal" not in text:
        text += """

from .vision_v4 import register_v4_multimodal

register_v4_multimodal(DeepseekV4ForCausalLM)
"""
    MODEL.write_text(text)
    print("patched model.py")


if __name__ == "__main__":
    main()
