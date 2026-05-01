"""
XGenDet Backbone: Frozen CLIP ViT-L/14 with Forgery Prompt Tokens and Layer Norm Tuning.

Key design:
- CLIP ViT-L/14 is completely frozen except LayerNorm parameters (LNCLIP-DF insight)
- 8 learnable Forgery Prompt Tokens are prepended to patch tokens before ViT processing
- Patch shuffling (D3 insight) creates distorted version for discrepancy detection
- Multi-layer attention maps are extracted for heatmap generation
"""

import sys
import os
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def _load_clip_module():
    """Load the OpenAI CLIP module, trying multiple paths."""
    # Try the standard OpenAI CLIP package
    try:
        import clip as _clip
        if hasattr(_clip, "load"):
            return _clip
    except ImportError:
        pass

    # Try D3's bundled CLIP (absolute import via importlib)
    d3_clip_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "GTA", "D3", "models", "clip")
    )
    if os.path.isdir(d3_clip_path):
        spec = importlib.util.spec_from_file_location(
            "d3_clip", os.path.join(d3_clip_path, "clip.py"),
            submodule_search_locations=[d3_clip_path],
        )
        mod = importlib.util.module_from_spec(spec)

        # Ensure sub-imports work: model.py and simple_tokenizer.py
        parent_dir = os.path.dirname(d3_clip_path)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Import as a package so relative imports resolve
        d3_models_clip = os.path.join(d3_clip_path, "__init__.py")
        pkg_spec = importlib.util.spec_from_file_location(
            "clip", d3_models_clip,
            submodule_search_locations=[d3_clip_path],
        )
        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        sys.modules["clip"] = pkg_mod

        # Now the submodules can resolve
        clip_spec = importlib.util.spec_from_file_location(
            "clip.clip", os.path.join(d3_clip_path, "clip.py"),
        )
        clip_mod = importlib.util.module_from_spec(clip_spec)
        sys.modules["clip.clip"] = clip_mod

        model_spec = importlib.util.spec_from_file_location(
            "clip.model", os.path.join(d3_clip_path, "model.py"),
        )
        model_mod = importlib.util.module_from_spec(model_spec)
        sys.modules["clip.model"] = model_mod

        tokenizer_spec = importlib.util.spec_from_file_location(
            "clip.simple_tokenizer", os.path.join(d3_clip_path, "simple_tokenizer.py"),
        )
        tokenizer_mod = importlib.util.module_from_spec(tokenizer_spec)
        sys.modules["clip.simple_tokenizer"] = tokenizer_mod

        # Execute in dependency order
        tokenizer_spec.loader.exec_module(tokenizer_mod)
        model_spec.loader.exec_module(model_mod)
        clip_spec.loader.exec_module(clip_mod)
        pkg_spec.loader.exec_module(pkg_mod)

        if hasattr(pkg_mod, "load"):
            return pkg_mod

    raise ImportError(
        "OpenAI CLIP not found. Install with: pip install git+https://github.com/openai/CLIP.git"
    )


clip = _load_clip_module()


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPBackboneWithPrompts(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_prompt_tokens: int = 8,
        tune_layer_norm: bool = True,
        extract_layers: tuple = (6, 12, 18, 23),
        adapter_blocks: list = None,   # e.g. list(range(12, 24)) for last 12 blocks
        adapter_bottleneck: int = 64,  # bottleneck dim per adapter
    ):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.num_prompt_tokens = num_prompt_tokens
        self.extract_layers = extract_layers
        self._adapter_blocks = adapter_blocks or []
        self._adapter_bottleneck = adapter_bottleneck

        # Load CLIP model
        self.model, self.preprocess = clip.load(clip_model_name, device="cpu")
        self.visual = self.model.visual

        # Get hidden dimension from CLIP model
        if "ViT-L" in clip_model_name:
            self.hidden_dim = 1024  # ViT-L internal dim
            self.output_dim = 768   # after projection
            self.num_layers = 24
            self.num_heads = 16
            if "336" in clip_model_name:
                # ViT-L/14@336px: patch_size=14, input=336 → 24×24 = 576 patches
                self.num_patches = 576
                self.grid_size = 24
            else:
                # ViT-L/14: patch_size=14, input=224 → 16×16 = 256 patches
                self.num_patches = 256
                self.grid_size = 16
        elif "ViT-B/16" in clip_model_name:
            self.hidden_dim = 768
            self.output_dim = 512
            self.num_layers = 12
            self.num_heads = 12
            self.num_patches = 196  # 14x14 grid
            self.grid_size = 14
        else:
            raise ValueError(f"Unsupported CLIP model: {clip_model_name}")

        # Forgery Prompt Tokens - learnable tokens prepended to ViT input
        self.forgery_prompts = nn.Parameter(
            torch.randn(num_prompt_tokens, self.hidden_dim) * 0.02
        )

        # Positional embeddings for prompt tokens (prevents spatial ambiguity)
        self.prompt_pos_embed = nn.Parameter(
            torch.randn(num_prompt_tokens, self.hidden_dim) * 0.02
        )

        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm parameters if requested (LNCLIP-DF insight)
        if tune_layer_norm:
            self._unfreeze_layer_norms()

        # Install ViT Adapter blocks if requested
        self.vit_adapters = None
        if self._adapter_blocks:
            from models.vit_adapters import CLIPViTWithAdapters
            self.vit_adapters = CLIPViTWithAdapters(
                transformer=self.visual.transformer,
                adapter_blocks=self._adapter_blocks,
                hidden_dim=self.hidden_dim,
                bottleneck_dim=self._adapter_bottleneck,
            )
            n_adap = self.vit_adapters.count_adapter_params()
            print(f"  [Backbone] Installed {len(self._adapter_blocks)} ViT adapters "
                  f"(bottleneck={self._adapter_bottleneck}, params={n_adap:,})")

        # Storage for attention maps (populated by hooks)
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self._register_attention_hooks()

        # Storage for pre-LN features (populated by hook)
        self.pre_ln_features: Optional[torch.Tensor] = None
        self._register_feature_hook()

    def _unfreeze_layer_norms(self):
        """Unfreeze only LayerNorm parameters in the visual encoder."""
        for name, param in self.visual.named_parameters():
            if "ln_" in name or "layer_norm" in name.lower():
                param.requires_grad = True

    def unfreeze_last_blocks(self, n: int = 2):
        """
        Unfreeze the last N transformer blocks of the CLIP visual encoder.
        Called AFTER __init__ when partial backbone fine-tuning is desired.
        LR for these params should be ~100x lower than head LR (set in get_trainable_params).
        """
        transformer = self.visual.transformer
        total_blocks = len(transformer.resblocks)
        for i in range(total_blocks - n, total_blocks):
            for param in transformer.resblocks[i].parameters():
                param.requires_grad = True
        print(f"  [Backbone] Unfroze last {n} of {total_blocks} transformer blocks "
              f"({sum(p.numel() for b in transformer.resblocks[-n:] for p in b.parameters()):,} params)")

    def _register_attention_hooks(self):
        """Register forward hooks on specified transformer layers to capture attention maps."""
        transformer = self.visual.transformer

        for layer_idx in self.extract_layers:
            if layer_idx < len(transformer.resblocks):
                block = transformer.resblocks[layer_idx]

                def make_hook(idx):
                    def hook_fn(module, input, output):
                        # Get attention weights from the multi-head attention
                        # input[0] shape: [seq_len, batch, hidden_dim]
                        x = input[0]
                        attn = module.attn
                        seq_len, batch_size, _ = x.shape

                        # Compute Q, K manually to get attention weights
                        x_ln = module.ln_1(x)
                        qkv = attn.in_proj_weight @ x_ln.reshape(-1, self.hidden_dim).T
                        qkv = qkv.T.reshape(seq_len, batch_size, 3, self.num_heads, self.hidden_dim // self.num_heads)
                        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                        # q, k: [seq_len, batch, heads, head_dim]
                        q = q.permute(1, 2, 0, 3)  # [batch, heads, seq_len, head_dim]
                        k = k.permute(1, 2, 0, 3)

                        scale = (self.hidden_dim // self.num_heads) ** -0.5
                        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attn_weights = F.softmax(attn_weights, dim=-1)

                        self.attention_maps[idx] = attn_weights.detach()

                    return hook_fn

                block.register_forward_hook(make_hook(layer_idx))

    def _register_feature_hook(self):
        """Register hook to capture features before the final projection (ln_post)."""
        def hook_fn(module, input, output):
            self.pre_ln_features = output.clone()

        self.visual.ln_post.register_forward_hook(hook_fn)

    def shuffle_patches(self, x: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
        """Shuffle image patches to create distorted version (D3 technique)."""
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        # patches: [B, C*patch_size*patch_size, num_patches]
        perm = torch.randperm(patches.size(-1), device=x.device)
        shuffled = patches[:, :, perm]
        shuffled_images = F.fold(
            shuffled, output_size=(H, W),
            kernel_size=patch_size, stride=patch_size
        )
        return shuffled_images

    def _inject_prompts_and_encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image through CLIP ViT with forgery prompt tokens injected.

        Returns:
            cls_token: [B, hidden_dim] - CLS token output
            patch_tokens: [B, num_patches, hidden_dim] - patch token outputs
        """
        visual = self.visual
        B = x.shape[0]

        # --- CLIP ViT forward with prompt injection ---
        # Step 1: Patch embedding
        x = visual.conv1(x)  # [B, hidden_dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, hidden_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, hidden_dim]

        # Step 2: Prepend CLS token
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+num_patches, hidden_dim]

        # Step 3: Add positional embeddings (only to CLS + patches)
        x = x + visual.positional_embedding.to(x.dtype)

        # Step 4: Inject forgery prompt tokens with their own positional embeddings
        prompts = self.forgery_prompts.unsqueeze(0).expand(B, -1, -1)  # [B, num_prompts, hidden_dim]
        prompts = prompts + self.prompt_pos_embed.unsqueeze(0)  # Add positional signal
        x = torch.cat([x[:, :1, :], prompts, x[:, 1:, :]], dim=1)
        # x: [B, 1+num_prompts+num_patches, hidden_dim]

        # Step 5: Permute for transformer (expects [seq_len, batch, hidden_dim])
        x = x.permute(1, 0, 2)  # [seq_len, B, hidden_dim]

        # Step 6: LayerNorm pre-transformer
        x = visual.ln_pre(x)

        # Step 7: Run through transformer blocks (with adapters if installed)
        if self.vit_adapters is not None:
            transformer_out = self.vit_adapters(x)
        else:
            transformer_out = visual.transformer(x)

        # Handle both standard CLIP (returns tensor) and D3 CLIP (returns tuple)
        if isinstance(transformer_out, tuple):
            x = transformer_out[-1]  # Last element is the feature tensor
        else:
            x = transformer_out

        # Step 8: Permute back
        x = x.permute(1, 0, 2)  # [B, seq_len, hidden_dim]

        # Step 9: Extract CLS and patch tokens (skip prompt tokens)
        cls_out = x[:, 0, :]  # [B, hidden_dim]
        patch_out = x[:, 1 + self.num_prompt_tokens:, :]  # [B, num_patches, hidden_dim]

        # Step 10: Apply ln_post to CLS (this triggers the feature hook)
        cls_out = visual.ln_post(cls_out)

        # Step 11: Project CLS to output dim
        if visual.proj is not None:
            cls_out = cls_out @ visual.proj  # [B, output_dim]

        return cls_out, patch_out

    def forward(
        self,
        x: torch.Tensor,
        return_shuffled: bool = True,
        shuffle_patch_size: int = 32,
    ) -> dict:
        """
        Forward pass through the backbone.

        Args:
            x: Input images [B, 3, 224, 224]
            return_shuffled: Whether to also encode shuffled version
            shuffle_patch_size: Patch size for shuffling

        Returns:
            dict with:
                - cls_orig: [B, output_dim] CLS token from original
                - patches_orig: [B, num_patches, hidden_dim] patch tokens from original
                - attn_maps_orig: dict of attention maps per layer
                - cls_shuf: [B, output_dim] CLS token from shuffled (if return_shuffled)
                - patches_shuf: [B, num_patches, hidden_dim] patch tokens from shuffled
                - attn_maps_shuf: dict of attention maps per layer
        """
        # Clear stored attention maps
        self.attention_maps = {}

        # Encode original image
        cls_orig, patches_orig = self._inject_prompts_and_encode(x)
        attn_maps_orig = dict(self.attention_maps)

        result = {
            "cls_orig": cls_orig,
            "patches_orig": patches_orig,
            "attn_maps_orig": attn_maps_orig,
        }

        if return_shuffled:
            # Clear and encode shuffled version
            self.attention_maps = {}
            with torch.no_grad():
                x_shuffled = self.shuffle_patches(x, patch_size=shuffle_patch_size)
            cls_shuf, patches_shuf = self._inject_prompts_and_encode(x_shuffled)
            attn_maps_shuf = dict(self.attention_maps)

            result.update({
                "cls_shuf": cls_shuf,
                "patches_shuf": patches_shuf,
                "attn_maps_shuf": attn_maps_shuf,
            })

        return result

    def get_trainable_params(self):
        """Return only trainable parameters for the optimizer."""
        params = []
        # Forgery prompt tokens + their positional embeddings
        params.append({"params": [self.forgery_prompts, self.prompt_pos_embed], "lr_scale": 1.0, "name": "forgery_prompts"})

        # LayerNorm parameters (lr_scale=0.01 relative to head)
        ln_params = [p for n, p in self.visual.named_parameters()
                     if p.requires_grad and ("ln_" in n or "layer_norm" in n.lower())]
        if ln_params:
            params.append({"params": ln_params, "lr_scale": 0.01, "name": "layer_norms"})

        # Unfrozen transformer block parameters (lr_scale=0.005 — very low, backbone fine-tune)
        block_params = [p for n, p in self.visual.transformer.named_parameters()
                        if p.requires_grad and "ln_" not in n]
        if block_params:
            params.append({"params": block_params, "lr_scale": 0.005, "name": "backbone_blocks"})

        # ViT Adapter parameters (lr_scale=0.2 — moderate, they are new random weights)
        if self.vit_adapters is not None:
            params.append({"params": list(self.vit_adapters.adapters.parameters()),
                           "lr_scale": 0.2, "name": "vit_adapters"})

        return params

    def count_trainable_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
