"""0731 视觉塔。前向与该检查点附带的 Vision-Exp 推理代码一致。

RoPE 把每个头的一半分给高、一半分给宽，不是相邻成对旋转。
Aligner 按通道在前的 unfold 做 3×3 合并。图像 token 使用 N 形排布
（两行交织，再按 4 对齐补齐），特征才能落到语言模型训练时阅读的位置。
图像 token 一律用词表内的 <｜image｜>（id 129279），不能用词表长度加类型号，
否则 hash MoE 的 tid2eid 会越界。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

PATCH = 14
MERGE = 3
MAX_TOKENS = 384
MIN_PIXELS = 147456
MAX_WH_RATIO = 8
COMPRESS_PAD_TO = 4

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * y).to(dtype)


def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """按行优先生成旋转角：先全部高度频率，再全部宽度频率。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """把头的后一半当作旋转的另一半，而不是两两相邻旋转。"""
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, inter: int, theta: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.theta = theta
        self.norm1 = RMSNorm(dim)
        self.attn = nn.Module()
        self.attn.wqkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn.wo = nn.Linear(dim, dim, bias=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Module()
        self.mlp.w1 = nn.Linear(dim, inter * 2, bias=False)
        self.mlp.w2 = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q, k, v = (t.view(x.size(0), self.heads, self.head_dim) for t in self.attn.wqkv(h).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        x = x + self.attn.wo(y.transpose(0, 1).reshape(x.size(0), -1))
        h = self.norm2(x)
        gate, up = self.mlp.w1(h).chunk(2, dim=-1)
        return x + self.mlp.w2(F.silu(gate) * up)


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = int(getattr(config, "vision_dim", 1024))
        layers = int(getattr(config, "vision_n_layers", 32))
        heads = int(getattr(config, "vision_n_heads", 16))
        inter = int(getattr(config, "vision_inter_dim", 2816))
        theta = float(getattr(config, "vision_rope_theta", 10000.0))
        self.rope_dim = dim // heads // 2
        self.rope_theta = theta
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(3 * PATCH * PATCH, dim, bias=True)
        self.blocks = nn.ModuleList([Block(dim, heads, inter, theta) for _ in range(layers)])
        self.norm = RMSNorm(dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed.proj(patches.flatten(1).to(self.patch_embed.proj.weight.dtype))
        cos, sin = vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """3×3 unfold。通道维在最前面，和官方实现一致，不能改成通道在最后。"""
    def __init__(self, vision_dim: int = 1024, hidden: int = 4096):
        super().__init__()
        self.downsample_ratio = MERGE
        self.w1 = nn.Linear(vision_dim * MERGE * MERGE, hidden, bias=True)
        self.w2 = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, (-n_w) % r, 0, (-n_h) % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x.to(self.w1.weight.dtype))))


def grid_tokens(height: int, width: int) -> tuple[int, int, int]:
    """估算一张图在语言模型里占多少位置，含换行和压缩补齐。"""
    n_llm_h = math.ceil((height // PATCH) / MERGE)
    n_llm_w = math.ceil((width // PATCH) / MERGE)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height: int, width: int, max_n_token: int):
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    cell = PATCH * MERGE
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * cell
        best_height = max_h * cell
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        best_width = max_w * cell
        best_height = max_h * cell
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * cell / width, max_h * cell / height)
        best_width = math.floor(width * beta / PATCH) * PATCH
        best_height = math.floor(height * beta / PATCH) * PATCH
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height: int, width: int, best_height: int, best_width: int, max_n_token: int):
    limit = max_n_token - (COMPRESS_PAD_TO - 1)
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width)
    budget = limit
    while num_tokens > limit and budget > 2:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(height, width, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def prepare_image(image, max_tokens: int = MAX_TOKENS, min_pixels: int = MIN_PIXELS, max_wh_ratio: int = MAX_WH_RATIO):
    """先补到 14 的倍数，不拉成 42 的倍数。像素按 (x/255-0.5)/0.5 归一化。"""
    """Resize one PIL image into ViT patches of shape (n_vit_h * n_vit_w, 3, 14, 14)."""
    from PIL import ImageOps

    image = image.convert("RGB")
    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
    best_width = math.ceil(width / PATCH) * PATCH
    best_height = math.ceil(height / PATCH) * PATCH
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, max_tokens
    )
    n_vit_h, n_vit_w = best_height // PATCH, best_width // PATCH
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    import numpy as np

    x = torch.from_numpy(np.asarray(image, dtype="float32")).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, PATCH, n_vit_w, PATCH).permute(1, 3, 0, 2, 4)
    patches = patches.reshape(n_vit_h * n_vit_w, 3, PATCH, PATCH).contiguous()
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 N 形图像块：起始位置补齐到模 4 余 3，两行交织，末尾补 IMAGE_END。

    start_pos 是展开后的绝对位置。补齐量因此依赖整段提示词里的位置，
    不能按图片哈希缓存这次替换。
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    body = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat([
        torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
        torch.tensor([IMAGE_START]),
        body[order],
        torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
        torch.tensor([IMAGE_END]),
    ])
    return types, perm


def _batched(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[0] == 1 and value.ndim > 1:
            return [value[0]]
        if value.ndim == 1:
            return [value]
        return list(value)
    return [value]


def encode_images(model, pixel_values, vision_grid=None) -> list:
    """每个占位符返回一行特征。IMAGE 槽写入 aligner 输出，特殊槽写入对应参数。"""
    pixels = _batched(pixel_values)
    grids = _batched(vision_grid)
    if not pixels:
        return []
    if len(pixels) != len(grids):
        raise RuntimeError(f"image tensors and grids disagree: {len(pixels)} vs {len(grids)}")

    device = model.image_start.device
    dtype = model.image_start.dtype
    specials = {
        IMAGE_START: model.image_start,
        IMAGE_PAD: model.image_pad,
        IMAGE_NEW_LINE: model.image_newline,
        IMAGE_END: model.image_end,
    }
    out = []
    for patches, grid in zip(pixels, grids):
        n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos = (int(v) for v in grid.detach().cpu().reshape(-1).tolist())
        patches = patches.to(device=device)
        if patches.ndim == 5:
            patches = patches[0]
        features = model.vision(patches, n_vit_h, n_vit_w)
        rows = model.aligner(features, n_vit_h, n_vit_w)
        types, perm = build_image_block(n_llm_h, n_llm_w, start_pos)
        if int((types == IMAGE).sum()) != rows.shape[0]:
            raise RuntimeError(
                f"aligner rows {rows.shape[0]} do not match image slots {int((types == IMAGE).sum())}"
            )
        block = rows.new_empty(types.shape[0], rows.shape[-1])
        types = types.to(device)
        for type_id, param in specials.items():
            mask = types == type_id
            if mask.any():
                block[mask] = param.to(dtype=block.dtype, device=device)
        block[types == IMAGE] = rows[perm.to(device)].to(block.dtype)
        out.append(block.to(dtype))
    return out


def register_v4_multimodal(model_cls) -> None:
    """向 vLLM 登记图像处理：缓存按绝对位置失效，整段占位都标成图像 token。"""
    from collections.abc import Mapping, Sequence

    from transformers import BatchFeature

    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.multimodal.parse import MultiModalDataItems
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
        PromptUpdateDetails,
    )

    class Info(BaseProcessingInfo):
        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"image": 4}

        def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
            return {"image": MAX_TOKENS}

    class Dummy(BaseDummyInputsBuilder[Info]):
        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            return "<｜image｜>" * int(mm_counts.get("image", 0))

        def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
            from PIL import Image

            n = int(mm_counts.get("image", 0))
            image = Image.new("RGB", (504, 672), (128, 128, 128))
            return {"image": [image] * n}

    class Processor(BaseMultiModalProcessor[Info]):
        def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
            return False

        def _cached_apply_hf_processor(self, inputs, timing_ctx):
            # Image-span padding depends on the placeholder's absolute position,
            # so a cache entry built for a different prompt cannot be reused.
            return self._apply_hf_processor(inputs, timing_ctx)

        def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
            tok = self.info.get_tokenizer()
            ids = tok.encode(prompt, add_special_tokens=False)
            images = mm_data.get("images") or mm_data.get("image") or []
            image_id = tok.convert_tokens_to_ids("<｜image｜>")
            patches, grids = [], []
            cursor = 0
            image_iter = iter(images)
            for tok_id in ids:
                if tok_id != image_id:
                    cursor += 1
                    continue
                prepared = prepare_image(next(image_iter))
                patch, n_vit_h, n_vit_w, n_llm_h, n_llm_w = prepared
                grids.append(torch.tensor([n_vit_h, n_vit_w, n_llm_h, n_llm_w, cursor], dtype=torch.int64))
                patches.append(patch)
                types, _perm = build_image_block(n_llm_h, n_llm_w, cursor)
                cursor += int(types.numel())
            return BatchFeature({"input_ids": [ids], "pixel_values": patches, "vision_grid": grids})

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            return {
                "pixel_values": MultiModalFieldConfig.batched("image"),
                "vision_grid": MultiModalFieldConfig.batched("image"),
            }

        def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[PromptUpdate]:
            image_id = self.info.get_tokenizer().convert_tokens_to_ids("<｜image｜>")
            items = list(out_mm_kwargs.get("image", []))

            def replacement(item_idx: int):
                grid = items[item_idx].get_data()["vision_grid"]
                _n_vit_h, _n_vit_w, n_llm_h, n_llm_w, start_pos = (int(v) for v in grid.reshape(-1).tolist())
                types, _perm = build_image_block(n_llm_h, n_llm_w, start_pos)
                tokens = [image_id] * int(types.numel())
                return PromptUpdateDetails.select_token_id(tokens, image_id)

            return [
                PromptReplacement(
                    modality="image",
                    target=[image_id],
                    replacement=replacement,
                )
            ]

    if getattr(model_cls, "_v4_mm_registered", False):
        return
    MULTIMODAL_REGISTRY.register_processor(Processor, info=Info, dummy_inputs=Dummy)(model_cls)
    model_cls._v4_mm_registered = True
