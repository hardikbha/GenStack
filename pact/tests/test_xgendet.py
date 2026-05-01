"""
Comprehensive test suite for XGenDet Stage 1 pipeline.

Run with:
    /home/sachin.chaudhary/.conda/envs/D3/bin/python -m pytest tests/test_xgendet.py -v
or:
    /home/sachin.chaudhary/.conda/envs/D3/bin/python -m unittest tests.test_xgendet -v
"""

import sys
import os
import unittest
import tempfile
import json

import numpy as np

sys.path.insert(0, '/home/sachin.chaudhary/xgendet')

import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Model Architecture Tests
# ---------------------------------------------------------------------------

class TestXGenDetModel(unittest.TestCase):
    """Tests for the full XGenDet model (backbone + prototype + heatmap + classification)."""

    @classmethod
    def setUpClass(cls):
        """Build the model once for all tests in this class."""
        from models.xgendet import XGenDet
        cls.device = torch.device("cpu")
        cls.model = XGenDet(
            clip_model_name="ViT-L/14",
            num_prompt_tokens=8,
            tune_layer_norm=True,
            num_prototypes=128,
            proto_dim=128,
            proto_heads=4,
            extract_layers=(6, 12, 18, 23),
            shuffle_patch_size=32,
            heatmap_output_size=224,
            num_families=4,
            dropout=0.2,
        ).to(cls.device)
        cls.batch_size = 2

    # -- instantiation -------------------------------------------------------

    def test_instantiation(self):
        """XGenDet should instantiate without error with default params."""
        self.assertIsInstance(self.model, nn.Module)

    # -- forward pass shapes -------------------------------------------------

    def test_forward_output_keys(self):
        """forward() should return all expected output keys."""
        x = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.model(x, return_heatmap=True)
        expected_keys = {
            "binary_logit", "confidence", "family_logit",
            "heatmap", "proto_activations", "proto_spatial_maps",
            "attr_scores", "proto_features", "patches_orig",
        }
        self.assertEqual(set(out.keys()), expected_keys)

    def test_forward_output_shapes(self):
        """All output tensors should have correct shapes."""
        B = self.batch_size
        x = torch.randn(B, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.model(x, return_heatmap=True)

        self.assertEqual(out["binary_logit"].shape, (B, 1))
        self.assertEqual(out["confidence"].shape, (B, 1))
        self.assertEqual(out["family_logit"].shape, (B, 4))
        self.assertEqual(out["heatmap"].shape, (B, 1, 224, 224))
        self.assertEqual(out["proto_activations"].shape, (B, 128))
        self.assertEqual(out["proto_spatial_maps"].shape, (B, 128, 16, 16))
        self.assertEqual(out["attr_scores"].shape, (B, 6))
        self.assertEqual(out["proto_features"].shape, (B, 128, 128))
        self.assertEqual(out["patches_orig"].shape, (B, 256, 1024))

    def test_forward_no_heatmap(self):
        """forward(return_heatmap=False) should return None for heatmap."""
        x = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.model(x, return_heatmap=False)
        self.assertIsNone(out["heatmap"])
        # Other outputs should still be present
        self.assertEqual(out["binary_logit"].shape, (self.batch_size, 1))

    # -- detect() inference method -------------------------------------------

    def test_detect_output_keys(self):
        """detect() should return inference-friendly outputs."""
        x = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        with torch.no_grad():
            result = self.model.detect(x)
        expected_keys = {
            "prediction", "confidence", "family",
            "heatmap", "attr_scores", "proto_activations",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_detect_output_shapes(self):
        """detect() outputs should have correct shapes."""
        B = self.batch_size
        x = torch.randn(B, 3, 224, 224, device=self.device)
        result = self.model.detect(x)
        self.assertEqual(result["prediction"].shape, (B,))
        self.assertEqual(result["confidence"].shape, (B,))
        self.assertEqual(result["family"].shape, (B,))
        self.assertEqual(result["heatmap"].shape, (B, 1, 224, 224))
        self.assertEqual(result["attr_scores"].shape, (B, 6))
        self.assertEqual(result["proto_activations"].shape, (B, 128))

    def test_detect_prediction_values(self):
        """detect() predictions should be 0 or 1."""
        x = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        result = self.model.detect(x)
        unique = result["prediction"].unique().tolist()
        for v in unique:
            self.assertIn(v, [0, 1])

    def test_detect_confidence_range(self):
        """detect() confidence should be in [0, 1]."""
        x = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        result = self.model.detect(x)
        self.assertTrue((result["confidence"] >= 0).all())
        self.assertTrue((result["confidence"] <= 1).all())

    # -- trainable parameter count -------------------------------------------

    def test_trainable_param_count(self):
        """Trainable parameter count should be in expected range (~1-2M)."""
        counts = self.model.count_trainable_params()
        total = counts["total"]
        # The trainable params should be roughly 1-3M (backbone LN + prompts + new heads)
        self.assertGreater(total, 500_000, f"Too few trainable params: {total}")
        self.assertLess(total, 5_000_000, f"Too many trainable params: {total}")

    def test_count_trainable_components(self):
        """count_trainable_params() should return all expected components."""
        counts = self.model.count_trainable_params()
        expected_keys = {"backbone", "prototype_module", "heatmap", "classification", "total"}
        self.assertEqual(set(counts.keys()), expected_keys)
        # Each component should have > 0 params
        for key in ["prototype_module", "heatmap", "classification"]:
            self.assertGreater(counts[key], 0, f"{key} has no trainable params")


class TestBackbone(unittest.TestCase):
    """Tests for CLIPBackboneWithPrompts."""

    @classmethod
    def setUpClass(cls):
        from models.backbone import CLIPBackboneWithPrompts
        cls.device = torch.device("cpu")
        cls.backbone = CLIPBackboneWithPrompts(
            clip_model_name="ViT-L/14",
            num_prompt_tokens=8,
            tune_layer_norm=True,
            extract_layers=(6, 12, 18, 23),
        ).to(cls.device)

    def test_backbone_output_keys(self):
        """Backbone forward should return expected keys."""
        x = torch.randn(2, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.backbone(x, return_shuffled=True, shuffle_patch_size=32)
        expected = {"cls_orig", "patches_orig", "attn_maps_orig",
                    "cls_shuf", "patches_shuf", "attn_maps_shuf"}
        self.assertEqual(set(out.keys()), expected)

    def test_backbone_output_shapes(self):
        """Backbone CLS and patch tokens should have correct shapes."""
        B = 2
        x = torch.randn(B, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.backbone(x, return_shuffled=True, shuffle_patch_size=32)
        # ViT-L/14: hidden_dim=1024, output_dim=768, num_patches=256
        self.assertEqual(out["cls_orig"].shape, (B, 768))
        self.assertEqual(out["patches_orig"].shape, (B, 256, 1024))
        self.assertEqual(out["cls_shuf"].shape, (B, 768))
        self.assertEqual(out["patches_shuf"].shape, (B, 256, 1024))

    def test_backbone_attention_maps(self):
        """Backbone should capture attention maps for specified layers."""
        x = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.backbone(x, return_shuffled=False)
        attn_maps = out["attn_maps_orig"]
        for layer_idx in (6, 12, 18, 23):
            self.assertIn(layer_idx, attn_maps, f"Missing attention map for layer {layer_idx}")
            # Shape: [B, heads, seq_len, seq_len]
            # seq_len = 1 (CLS) + 8 (prompts) + 256 (patches) = 265
            attn = attn_maps[layer_idx]
            self.assertEqual(attn.ndim, 4)
            self.assertEqual(attn.shape[0], 1)   # batch
            self.assertEqual(attn.shape[1], 16)   # heads

    def test_backbone_no_shuffled(self):
        """Backbone with return_shuffled=False should not have shuf keys."""
        x = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.backbone(x, return_shuffled=False)
        self.assertNotIn("cls_shuf", out)

    def test_backbone_frozen_params(self):
        """Most backbone params should be frozen (requires_grad=False)."""
        total = sum(1 for p in self.backbone.model.parameters())
        frozen = sum(1 for p in self.backbone.model.parameters() if not p.requires_grad)
        # Majority should be frozen; with LayerNorm tuning enabled the frozen
        # fraction is ~77% (by parameter count the frozen fraction by numel is
        # much higher). We check that at least 70% of parameter tensors are frozen.
        self.assertGreater(frozen / total, 0.7,
                           "Less than 70% of CLIP parameter tensors are frozen")

    def test_forgery_prompts_trainable(self):
        """Forgery prompt tokens should be trainable."""
        self.assertTrue(self.backbone.forgery_prompts.requires_grad)
        self.assertEqual(self.backbone.forgery_prompts.shape, (8, 1024))

    def test_get_trainable_params(self):
        """get_trainable_params() should return param groups with expected names."""
        groups = self.backbone.get_trainable_params()
        names = {g["name"] for g in groups}
        self.assertIn("forgery_prompts", names)
        self.assertIn("layer_norms", names)


class TestPrototypeModule(unittest.TestCase):
    """Tests for the PrototypeModule (PGAD)."""

    @classmethod
    def setUpClass(cls):
        from models.prototype_module import PrototypeModule, NUM_ATTRIBUTES, ATTRIBUTE_BANKS
        cls.device = torch.device("cpu")
        cls.module = PrototypeModule(
            input_dim=1024,
            proto_dim=128,
            num_prototypes=128,
            num_heads=4,
            grid_size=16,
        ).to(cls.device)
        cls.NUM_ATTRIBUTES = NUM_ATTRIBUTES
        cls.ATTRIBUTE_BANKS = ATTRIBUTE_BANKS

    def test_forward_output_shapes(self):
        """Prototype module should return correct shapes for all outputs."""
        B = 2
        patches = torch.randn(B, 256, 1024, device=self.device)
        proto_act, proto_spatial, attr_scores, proto_feats = self.module(patches)
        self.assertEqual(proto_act.shape, (B, 128))
        self.assertEqual(proto_spatial.shape, (B, 128, 16, 16))
        self.assertEqual(attr_scores.shape, (B, self.NUM_ATTRIBUTES))
        self.assertEqual(proto_feats.shape, (B, 128, 128))

    def test_proto_activations_range(self):
        """Proto activations should be in [0, 1] (sigmoid output)."""
        patches = torch.randn(1, 256, 1024, device=self.device)
        proto_act, _, _, _ = self.module(patches)
        self.assertTrue((proto_act >= 0).all())
        self.assertTrue((proto_act <= 1).all())

    def test_attribute_banks_coverage(self):
        """Attribute banks should cover all 128 prototypes without overlap."""
        covered = set()
        for name, (start, end) in self.ATTRIBUTE_BANKS.items():
            for i in range(start, end):
                self.assertNotIn(i, covered, f"Overlap at index {i}")
                covered.add(i)
        self.assertEqual(len(covered), 128)

    def test_num_attributes(self):
        """NUM_ATTRIBUTES should equal 6."""
        self.assertEqual(self.NUM_ATTRIBUTES, 6)

    def test_diversity_loss(self):
        """prototype_diversity_loss() should return a scalar."""
        loss = self.module.prototype_diversity_loss()
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_compactness_loss(self):
        """prototype_compactness_loss() should return a scalar."""
        patches = torch.randn(2, 256, 1024, device=self.device)
        loss = self.module.prototype_compactness_loss(patches)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_get_top_prototypes(self):
        """get_top_prototypes() should return structured results."""
        proto_act = torch.rand(2, 128, device=self.device)
        results = self.module.get_top_prototypes(proto_act, top_k=5)
        self.assertEqual(len(results), 2)
        for sample in results:
            self.assertEqual(len(sample), 5)
            for entry in sample:
                self.assertIn("prototype_id", entry)
                self.assertIn("activation", entry)
                self.assertIn("attribute", entry)
                self.assertIn(entry["attribute"],
                              list(self.ATTRIBUTE_BANKS.keys()) + ["unknown"])


class TestHeatmapGenerator(unittest.TestCase):
    """Tests for HeatmapGenerator and HeatmapStatistics."""

    @classmethod
    def setUpClass(cls):
        from models.heatmap_generator import HeatmapGenerator, HeatmapStatistics
        cls.device = torch.device("cpu")
        cls.generator = HeatmapGenerator(
            extract_layers=(6, 12, 18, 23),
            grid_size=16,
            output_size=224,
            num_prompt_tokens=8,
        ).to(cls.device)
        cls.stats = HeatmapStatistics(output_dim=32).to(cls.device)

    def _make_attn_maps(self, B, seq_len=265):
        """Create synthetic attention maps matching CLIP output format."""
        maps = {}
        for layer_idx in (6, 12, 18, 23):
            # [B, heads=16, seq_len, seq_len]
            attn = torch.softmax(torch.randn(B, 16, seq_len, seq_len, device=self.device), dim=-1)
            maps[layer_idx] = attn
        return maps

    def test_heatmap_output_shape(self):
        """Heatmap generator should output [B, 1, 224, 224]."""
        B = 2
        attn_orig = self._make_attn_maps(B)
        attn_shuf = self._make_attn_maps(B)
        proto_act = torch.rand(B, 128, device=self.device)
        proto_spatial = torch.rand(B, 128, 16, 16, device=self.device)

        heatmap = self.generator(attn_orig, attn_shuf, proto_act, proto_spatial)
        self.assertEqual(heatmap.shape, (B, 1, 224, 224))

    def test_heatmap_value_range(self):
        """Heatmap values should be in [0, 1] (sigmoid output)."""
        B = 2
        attn_orig = self._make_attn_maps(B)
        attn_shuf = self._make_attn_maps(B)
        proto_act = torch.rand(B, 128, device=self.device)
        proto_spatial = torch.rand(B, 128, 16, 16, device=self.device)

        heatmap = self.generator(attn_orig, attn_shuf, proto_act, proto_spatial)
        self.assertTrue((heatmap >= 0).all())
        self.assertTrue((heatmap <= 1).all())

    def test_heatmap_no_shuffled(self):
        """Heatmap generator should handle None for shuffled attention."""
        B = 2
        attn_orig = self._make_attn_maps(B)
        proto_act = torch.rand(B, 128, device=self.device)
        proto_spatial = torch.rand(B, 128, 16, 16, device=self.device)

        heatmap = self.generator(attn_orig, None, proto_act, proto_spatial)
        self.assertEqual(heatmap.shape, (B, 1, 224, 224))

    def test_heatmap_statistics_shape(self):
        """HeatmapStatistics should output [B, 32]."""
        B = 2
        heatmap = torch.rand(B, 1, 224, 224, device=self.device)
        stats = self.stats(heatmap)
        self.assertEqual(stats.shape, (B, 32))


class TestClassificationHead(unittest.TestCase):
    """Tests for the ClassificationHead."""

    @classmethod
    def setUpClass(cls):
        from models.classification_head import ClassificationHead
        cls.device = torch.device("cpu")
        cls.head = ClassificationHead(
            cls_dim=768,
            proto_dim=128,
            heatmap_stat_dim=32,
            num_attributes=6,
            num_families=4,
            dropout=0.2,
        ).to(cls.device)

    def test_forward_output_keys(self):
        """ClassificationHead should return expected keys."""
        B = 2
        out = self.head(
            cls_orig=torch.randn(B, 768, device=self.device),
            cls_shuf=torch.randn(B, 768, device=self.device),
            proto_features=torch.randn(B, 128, 128, device=self.device),
            proto_activations=torch.rand(B, 128, device=self.device),
            heatmap_stats=torch.randn(B, 32, device=self.device),
            attr_scores=torch.randn(B, 6, device=self.device),
        )
        self.assertEqual(set(out.keys()), {"binary_logit", "confidence", "family_logit"})

    def test_forward_output_shapes(self):
        """ClassificationHead outputs should have correct shapes."""
        B = 4
        out = self.head(
            cls_orig=torch.randn(B, 768, device=self.device),
            cls_shuf=torch.randn(B, 768, device=self.device),
            proto_features=torch.randn(B, 128, 128, device=self.device),
            proto_activations=torch.rand(B, 128, device=self.device),
            heatmap_stats=torch.randn(B, 32, device=self.device),
            attr_scores=torch.randn(B, 6, device=self.device),
        )
        self.assertEqual(out["binary_logit"].shape, (B, 1))
        self.assertEqual(out["confidence"].shape, (B, 1))
        self.assertEqual(out["family_logit"].shape, (B, 4))

    def test_confidence_range(self):
        """confidence should be in [0, 1] (sigmoid output)."""
        B = 2
        out = self.head(
            cls_orig=torch.randn(B, 768, device=self.device),
            cls_shuf=torch.randn(B, 768, device=self.device),
            proto_features=torch.randn(B, 128, 128, device=self.device),
            proto_activations=torch.rand(B, 128, device=self.device),
            heatmap_stats=torch.randn(B, 32, device=self.device),
            attr_scores=torch.randn(B, 6, device=self.device),
        )
        self.assertTrue((out["confidence"] >= 0).all())
        self.assertTrue((out["confidence"] <= 1).all())

    def test_prototype_weighted_pool(self):
        """prototype_weighted_pool should output [B, proto_dim]."""
        B = 2
        proto_features = torch.randn(B, 128, 128, device=self.device)
        proto_activations = torch.rand(B, 128, device=self.device)
        pooled = self.head.prototype_weighted_pool(proto_features, proto_activations)
        self.assertEqual(pooled.shape, (B, 128))


# ---------------------------------------------------------------------------
# 2. Training Component Tests
# ---------------------------------------------------------------------------

class TestXGenDetLoss(unittest.TestCase):
    """Tests for XGenDetLoss."""

    @classmethod
    def setUpClass(cls):
        from training.losses import XGenDetLoss
        from models.prototype_module import PrototypeModule
        cls.device = torch.device("cpu")
        cls.loss_fn = XGenDetLoss(
            w_family=0.5,
            w_proto_div=0.3,
            w_proto_compact=0.2,
            w_heatmap=0.3,
            w_attr=0.1,
            w_calib=0.2,
        ).to(cls.device)
        cls.proto_module = PrototypeModule(
            input_dim=1024, proto_dim=128,
            num_prototypes=128, num_heads=4, grid_size=16,
        ).to(cls.device)

    def _make_outputs(self, B=4):
        """Create synthetic model outputs for loss computation."""
        return {
            "binary_logit": torch.randn(B, 1, device=self.device),
            "confidence": torch.sigmoid(torch.randn(B, 1, device=self.device)),
            "family_logit": torch.randn(B, 4, device=self.device),
            "heatmap": torch.sigmoid(torch.randn(B, 1, 224, 224, device=self.device)),
            "proto_activations": torch.sigmoid(torch.randn(B, 128, device=self.device)),
            "proto_spatial_maps": torch.rand(B, 128, 16, 16, device=self.device),
            "attr_scores": torch.sigmoid(torch.randn(B, 6, device=self.device)),
            "proto_features": torch.randn(B, 128, 128, device=self.device),
            "patches_orig": torch.randn(B, 256, 1024, device=self.device),
        }

    def test_loss_returns_all_7_components(self):
        """Loss forward should return all 7 individual losses + total."""
        B = 4
        outputs = self._make_outputs(B)
        labels = torch.randint(0, 2, (B,), device=self.device)
        family_labels = torch.randint(0, 4, (B,), device=self.device)

        losses = self.loss_fn(outputs, labels, family_labels,
                              prototype_module=self.proto_module)

        expected_keys = {"cls", "family", "proto_div", "proto_compact",
                         "heatmap", "attr", "calib", "total"}
        self.assertEqual(set(losses.keys()), expected_keys)

        for key, val in losses.items():
            self.assertEqual(val.ndim, 0, f"{key} loss is not scalar")
            self.assertTrue(torch.isfinite(val), f"{key} loss is not finite")

    def test_loss_without_prototype_module(self):
        """Loss should work when prototype_module is None (zeros for proto losses)."""
        B = 4
        outputs = self._make_outputs(B)
        labels = torch.randint(0, 2, (B,), device=self.device)
        family_labels = torch.randint(0, 4, (B,), device=self.device)

        losses = self.loss_fn(outputs, labels, family_labels, prototype_module=None)
        self.assertEqual(losses["proto_div"].item(), 0.0)
        self.assertEqual(losses["proto_compact"].item(), 0.0)

    def test_loss_backward(self):
        """Total loss should be differentiable (gradients flow)."""
        B = 4
        # Use tensors that require grad
        outputs = {
            "binary_logit": torch.randn(B, 1, device=self.device, requires_grad=True),
            "confidence": torch.sigmoid(torch.randn(B, 1, device=self.device, requires_grad=True)),
            "family_logit": torch.randn(B, 4, device=self.device, requires_grad=True),
            "heatmap": torch.sigmoid(torch.randn(B, 1, 56, 56, device=self.device, requires_grad=True)),
            "proto_activations": torch.sigmoid(torch.randn(B, 128, device=self.device)),
            "proto_spatial_maps": torch.rand(B, 128, 16, 16, device=self.device),
            "attr_scores": torch.sigmoid(torch.randn(B, 6, device=self.device, requires_grad=True)),
            "proto_features": torch.randn(B, 128, 128, device=self.device),
            "patches_orig": torch.randn(B, 256, 1024, device=self.device),
        }
        labels = torch.randint(0, 2, (B,), device=self.device)
        family_labels = torch.randint(0, 4, (B,), device=self.device)

        losses = self.loss_fn(outputs, labels, family_labels,
                              prototype_module=self.proto_module)
        losses["total"].backward()

        self.assertIsNotNone(outputs["binary_logit"].grad)
        self.assertFalse(torch.all(outputs["binary_logit"].grad == 0))

    def test_heatmap_loss_none(self):
        """heatmap_loss should return 0 when heatmap is None."""
        labels = torch.randint(0, 2, (4,), device=self.device)
        loss = self.loss_fn.heatmap_loss(None, labels)
        self.assertEqual(loss.item(), 0.0)

    def test_classification_loss_shape(self):
        """classification_loss should return a scalar."""
        logit = torch.randn(4, 1, device=self.device)
        labels = torch.randint(0, 2, (4,), device=self.device)
        loss = self.loss_fn.classification_loss(logit, labels)
        self.assertEqual(loss.ndim, 0)

    def test_family_loss_shape(self):
        """family_loss should return a scalar."""
        logit = torch.randn(4, 4, device=self.device)
        labels = torch.randint(0, 4, (4,), device=self.device)
        loss = self.loss_fn.family_loss(logit, labels)
        self.assertEqual(loss.ndim, 0)

    def test_attribute_loss_real_only(self):
        """Attribute loss with all-real labels should penalise high attr_scores."""
        attr_high = torch.ones(4, 6, device=self.device)
        labels_real = torch.zeros(4, device=self.device, dtype=torch.long)
        loss_high = self.loss_fn.attribute_loss(attr_high, labels_real)

        attr_low = torch.zeros(4, 6, device=self.device)
        loss_low = self.loss_fn.attribute_loss(attr_low, labels_real)

        self.assertGreater(loss_high.item(), loss_low.item())


class TestGetTrainableParams(unittest.TestCase):
    """Tests for get_trainable_params() on the full model."""

    @classmethod
    def setUpClass(cls):
        from models.xgendet import XGenDet
        cls.device = torch.device("cpu")
        cls.model = XGenDet().to(cls.device)

    def test_returns_list_of_dicts(self):
        groups = self.model.get_trainable_params()
        self.assertIsInstance(groups, list)
        for g in groups:
            self.assertIn("params", g)
            self.assertIn("name", g)

    def test_all_groups_have_params(self):
        """Every param group should contain at least one parameter."""
        groups = self.model.get_trainable_params()
        for g in groups:
            self.assertGreater(len(g["params"]), 0,
                               f"Group '{g['name']}' has no params")

    def test_expected_group_names(self):
        """Should have groups for prompts, layer norms, prototype, heatmap, classification."""
        groups = self.model.get_trainable_params()
        names = {g["name"] for g in groups}
        for expected in ["forgery_prompts", "layer_norms",
                         "prototype_module", "heatmap", "classification"]:
            self.assertIn(expected, names)


class TestTemperatureScaling(unittest.TestCase):
    """Tests for TemperatureScaling and compute_ece."""

    def test_temperature_scaling_init(self):
        from training.calibration import TemperatureScaling
        ts = TemperatureScaling()
        self.assertEqual(ts.temperature, 1.0)

    def test_calibrate_with_default_temperature(self):
        """calibrate() with temp=1 should be equivalent to plain sigmoid."""
        from training.calibration import TemperatureScaling
        ts = TemperatureScaling()
        logits = torch.tensor([0.0, 1.0, -1.0])
        calibrated = ts.calibrate(logits)
        expected = torch.sigmoid(logits)
        self.assertTrue(torch.allclose(calibrated, expected, atol=1e-6))

    def test_calibrate_with_custom_temperature(self):
        """calibrate() with temp!=1 should scale logits before sigmoid."""
        from training.calibration import TemperatureScaling
        ts = TemperatureScaling()
        ts.temperature = 2.0
        logits = torch.tensor([0.0, 2.0, -2.0])
        calibrated = ts.calibrate(logits)
        expected = torch.sigmoid(logits / 2.0)
        self.assertTrue(torch.allclose(calibrated, expected, atol=1e-6))

    def test_compute_ece_perfect_calibration(self):
        """ECE should be ~0 when confidences match accuracies perfectly."""
        from training.calibration import compute_ece
        # Perfect calibration: 100 samples at confidence ~0.9, all correct
        confidences = np.full(100, 0.9)
        accuracies = np.ones(100)
        ece = compute_ece(confidences, accuracies, n_bins=15)
        self.assertLess(ece, 0.15)  # Should be very low

    def test_compute_ece_poor_calibration(self):
        """ECE should be high when confidences are far from accuracies."""
        from training.calibration import compute_ece
        # Overconfident: confidence=0.9 but accuracy=0 (all wrong)
        confidences = np.full(100, 0.9)
        accuracies = np.zeros(100)
        ece = compute_ece(confidences, accuracies, n_bins=15)
        self.assertGreater(ece, 0.5)

    def test_compute_ece_bounds(self):
        """ECE should always be in [0, 1]."""
        from training.calibration import compute_ece
        rng = np.random.RandomState(42)
        confidences = rng.uniform(0, 1, 200)
        accuracies = rng.randint(0, 2, 200).astype(float)
        ece = compute_ece(confidences, accuracies, n_bins=15)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


# ---------------------------------------------------------------------------
# 3. Data Pipeline Tests
# ---------------------------------------------------------------------------

class TestAugmentations(unittest.TestCase):
    """Tests for data augmentation transforms."""

    def test_get_train_transforms_returns_compose(self):
        from data.augmentations import get_train_transforms
        import torchvision.transforms as T
        t = get_train_transforms(crop_size=224, jpeg_prob=0.5, blur_prob=0.5)
        self.assertIsInstance(t, T.Compose)

    def test_get_eval_transforms_returns_compose(self):
        from data.augmentations import get_eval_transforms
        import torchvision.transforms as T
        t = get_eval_transforms(crop_size=224)
        self.assertIsInstance(t, T.Compose)

    def test_train_transform_output_shape(self):
        """Train transform should produce a [3, 224, 224] tensor."""
        from data.augmentations import get_train_transforms
        t = get_train_transforms(crop_size=224, jpeg_prob=0.0, blur_prob=0.0)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        out = t(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_eval_transform_output_shape(self):
        """Eval transform should produce a [3, 224, 224] tensor."""
        from data.augmentations import get_eval_transforms
        t = get_eval_transforms(crop_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        out = t(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_transform_value_range(self):
        """After CLIP normalization, values should not be raw [0,1]."""
        from data.augmentations import get_eval_transforms
        t = get_eval_transforms(crop_size=224)
        # All-white image
        img = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))
        out = t(img)
        # Normalized values should not all be in [0, 1] because CLIP normalization
        # shifts values so some channel values will exceed 1
        self.assertTrue(out.max() > 1.0 or out.min() < 0.0,
                        "Normalization does not seem applied")

    def test_train_transform_with_augmentations(self):
        """Train transform with JPEG/blur enabled should still produce valid output."""
        from data.augmentations import get_train_transforms
        t = get_train_transforms(crop_size=224, jpeg_prob=1.0, blur_prob=1.0)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        out = t(img)
        self.assertEqual(out.shape, (3, 224, 224))
        self.assertTrue(torch.isfinite(out).all())

    def test_custom_crop_size(self):
        """Transforms should respect custom crop_size."""
        from data.augmentations import get_eval_transforms
        t = get_eval_transforms(crop_size=128)
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        out = t(img)
        self.assertEqual(out.shape, (3, 128, 128))


class TestDatasetConstants(unittest.TestCase):
    """Tests for dataset constants and utility functions."""

    def test_generator_families_has_expected_keys(self):
        """GENERATOR_FAMILIES should contain known generators."""
        from data.dataset import GENERATOR_FAMILIES
        expected_generators = ["progan", "stylegan", "ADM", "GLIDE",
                               "Midjourney", "stable_diffusion"]
        for gen in expected_generators:
            self.assertIn(gen, GENERATOR_FAMILIES,
                          f"Missing generator: {gen}")

    def test_generator_families_values(self):
        """GENERATOR_FAMILIES values should be 1, 2, or 3."""
        from data.dataset import GENERATOR_FAMILIES
        valid_families = {1, 2, 3}
        for gen, family in GENERATOR_FAMILIES.items():
            self.assertIn(family, valid_families,
                          f"Invalid family {family} for {gen}")

    def test_infer_generator_family_known(self):
        """infer_generator_family should return correct family for known generators."""
        from data.dataset import infer_generator_family
        self.assertEqual(infer_generator_family("ProGAN"), 1)
        self.assertEqual(infer_generator_family("stylegan2"), 1)
        self.assertEqual(infer_generator_family("ADM"), 2)
        self.assertEqual(infer_generator_family("Midjourney"), 3)

    def test_infer_generator_family_unknown(self):
        """infer_generator_family should return 0 for unknown generators."""
        from data.dataset import infer_generator_family
        self.assertEqual(infer_generator_family("totally_unknown_gen"), 0)

    def test_image_extensions(self):
        """IMAGE_EXTENSIONS should include common formats."""
        from data.dataset import IMAGE_EXTENSIONS
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            self.assertIn(ext, IMAGE_EXTENSIONS)


# ---------------------------------------------------------------------------
# 4. MLLM Module Tests
# ---------------------------------------------------------------------------

class TestMLLMParsing(unittest.TestCase):
    """Tests for parse_mllm_output() -- no model loading needed."""

    def test_parse_valid_response(self):
        """parse_mllm_output should extract attributes and explanation from well-formed text."""
        from models.mllm_module import parse_mllm_output, ATTRIBUTE_NAMES

        text = """<attributes>
texture_consistency: 0.85
edge_quality: 0.72
color_distribution: 0.60
geometric_coherence: 0.45
semantic_plausibility: 0.90
frequency_artifacts: 0.30
</attributes>
<explanation>
The image shows clear signs of AI generation with noticeable edge artifacts.
</explanation>"""

        result = parse_mllm_output(text)
        self.assertIn("attributes", result)
        self.assertIn("explanation", result)

        attrs = result["attributes"]
        self.assertEqual(len(attrs), 6)
        self.assertAlmostEqual(attrs["texture_consistency"], 0.85, places=2)
        self.assertAlmostEqual(attrs["edge_quality"], 0.72, places=2)
        self.assertAlmostEqual(attrs["frequency_artifacts"], 0.30, places=2)

        self.assertIn("edge artifacts", result["explanation"])

    def test_parse_malformed_no_attributes(self):
        """parse_mllm_output should handle missing <attributes> block gracefully."""
        from models.mllm_module import parse_mllm_output

        text = "This is just some text without any structured format."
        result = parse_mllm_output(text)
        self.assertIsInstance(result["attributes"], dict)
        # attributes should be empty when no block found
        self.assertEqual(len(result["attributes"]), 0)
        # Explanation should fall back to the raw text
        self.assertEqual(result["explanation"], text.strip())

    def test_parse_malformed_partial_attributes(self):
        """parse_mllm_output should default missing attrs to 0.5."""
        from models.mllm_module import parse_mllm_output

        text = """<attributes>
texture_consistency: 0.9
edge_quality: 0.8
</attributes>
<explanation>
Partial response.
</explanation>"""

        result = parse_mllm_output(text)
        attrs = result["attributes"]
        self.assertAlmostEqual(attrs["texture_consistency"], 0.9, places=2)
        self.assertAlmostEqual(attrs["edge_quality"], 0.8, places=2)
        # Missing attributes should default to 0.5
        self.assertAlmostEqual(attrs["color_distribution"], 0.5, places=2)
        self.assertAlmostEqual(attrs["frequency_artifacts"], 0.5, places=2)

    def test_parse_malformed_no_explanation(self):
        """parse_mllm_output should handle missing <explanation> block."""
        from models.mllm_module import parse_mllm_output

        text = """<attributes>
texture_consistency: 0.5
edge_quality: 0.5
color_distribution: 0.5
geometric_coherence: 0.5
semantic_plausibility: 0.5
frequency_artifacts: 0.5
</attributes>
Some trailing text that should become the explanation."""

        result = parse_mllm_output(text)
        self.assertIn("trailing text", result["explanation"])


class TestMLLMExplainerInit(unittest.TestCase):
    """Tests for MLLMExplainer instantiation (without loading the actual model)."""

    def test_instantiation_defaults(self):
        """MLLMExplainer should instantiate without loading model."""
        from models.mllm_module import MLLMExplainer
        explainer = MLLMExplainer()
        self.assertIsNone(explainer.model)
        self.assertIsNone(explainer.processor)
        self.assertEqual(explainer.model_name, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertTrue(explainer.use_lora)

    def test_instantiation_custom_params(self):
        """MLLMExplainer should accept custom parameters."""
        from models.mllm_module import MLLMExplainer
        explainer = MLLMExplainer(
            model_name="custom/model",
            use_lora=False,
            lora_r=32,
            max_new_tokens=256,
        )
        self.assertEqual(explainer.model_name, "custom/model")
        self.assertFalse(explainer.use_lora)
        self.assertEqual(explainer.max_new_tokens, 256)

    def test_build_prompt(self):
        """_build_prompt should produce a well-formed prompt string."""
        from models.mllm_module import MLLMExplainer
        explainer = MLLMExplainer()
        stage1_outputs = {
            "confidence": 0.95,
            "family": 2,
            "attr_scores": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        }
        prompt = explainer._build_prompt(stage1_outputs)
        self.assertIn("FAKE", prompt)
        self.assertIn("Diffusion", prompt)
        self.assertIn("95.0%", prompt)

    def test_build_prompt_real(self):
        """_build_prompt should say REAL when confidence <= 0.5."""
        from models.mllm_module import MLLMExplainer
        explainer = MLLMExplainer()
        stage1_outputs = {
            "confidence": 0.3,
            "family": 0,
            "attr_scores": [0.1] * 6,
        }
        prompt = explainer._build_prompt(stage1_outputs)
        self.assertIn("REAL", prompt)


class TestMLLMConstants(unittest.TestCase):
    """Tests for MLLM module constants."""

    def test_attribute_names(self):
        from models.mllm_module import ATTRIBUTE_NAMES
        self.assertEqual(len(ATTRIBUTE_NAMES), 6)
        self.assertIn("texture_consistency", ATTRIBUTE_NAMES)

    def test_family_names(self):
        from models.mllm_module import FAMILY_NAMES
        self.assertEqual(len(FAMILY_NAMES), 4)
        self.assertEqual(FAMILY_NAMES[0], "Real")


# ---------------------------------------------------------------------------
# 5. XGenBench Tests
# ---------------------------------------------------------------------------

class TestXGenBenchDetection(unittest.TestCase):
    """Tests for XGenBench.evaluate_detection() with synthetic data."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary annotation file for XGenBench."""
        cls.tmpdir = tempfile.mkdtemp()
        cls.ann_file = os.path.join(cls.tmpdir, "annotations.jsonl")

        # 20 real, 20 fake
        annotations = []
        for i in range(20):
            annotations.append({
                "image_path": f"real_{i}.png",
                "label": 0,
                "attributes": {
                    "texture_consistency": 0.1,
                    "edge_quality": 0.1,
                    "color_distribution": 0.1,
                    "geometric_coherence": 0.1,
                    "semantic_plausibility": 0.1,
                    "frequency_artifacts": 0.1,
                },
            })
        for i in range(20):
            annotations.append({
                "image_path": f"fake_{i}.png",
                "label": 1,
                "attributes": {
                    "texture_consistency": 0.9,
                    "edge_quality": 0.8,
                    "color_distribution": 0.7,
                    "geometric_coherence": 0.6,
                    "semantic_plausibility": 0.5,
                    "frequency_artifacts": 0.9,
                },
            })

        with open(cls.ann_file, "w") as f:
            for entry in annotations:
                f.write(json.dumps(entry) + "\n")

        from data.xgenbench import XGenBench
        cls.bench = XGenBench(
            annotation_file=cls.ann_file,
            image_root=cls.tmpdir,
        )

    def test_evaluate_detection_perfect(self):
        """Perfect predictions should yield AP and AUC close to 1.0."""
        predictions = []
        for i in range(20):
            predictions.append({"image_path": f"real_{i}.png", "confidence": 0.1})
        for i in range(20):
            predictions.append({"image_path": f"fake_{i}.png", "confidence": 0.9})

        results = self.bench.evaluate_detection(predictions)
        self.assertIn("AP", results)
        self.assertIn("AUC", results)
        self.assertIn("Acc@0.5", results)
        self.assertIn("F1", results)
        self.assertGreater(results["AP"], 0.95)
        self.assertGreater(results["AUC"], 0.95)
        self.assertGreater(results["Acc@0.5"], 0.95)

    def test_evaluate_detection_random(self):
        """Random predictions should yield metrics around 0.5."""
        rng = np.random.RandomState(42)
        predictions = []
        for i in range(20):
            predictions.append({"image_path": f"real_{i}.png",
                                "confidence": float(rng.uniform(0, 1))})
        for i in range(20):
            predictions.append({"image_path": f"fake_{i}.png",
                                "confidence": float(rng.uniform(0, 1))})

        results = self.bench.evaluate_detection(predictions)
        # Random should give roughly 0.5 AUC
        self.assertGreater(results["AUC"], 0.2)
        self.assertLess(results["AUC"], 0.8)

    def test_evaluate_detection_empty(self):
        """Empty predictions should return empty dict."""
        results = self.bench.evaluate_detection([])
        self.assertEqual(results, {})

    def test_evaluate_detection_partial(self):
        """Predictions for only some images should work correctly."""
        predictions = []
        for i in range(10):
            predictions.append({"image_path": f"real_{i}.png", "confidence": 0.1})
        for i in range(10):
            predictions.append({"image_path": f"fake_{i}.png", "confidence": 0.9})

        results = self.bench.evaluate_detection(predictions)
        self.assertIn("AP", results)
        self.assertGreater(results["AP"], 0.9)


class TestXGenBenchAttributes(unittest.TestCase):
    """Tests for XGenBench.evaluate_attributes() with synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.ann_file = os.path.join(cls.tmpdir, "annotations.jsonl")

        annotations = []
        for i in range(30):
            annotations.append({
                "image_path": f"img_{i}.png",
                "label": i % 2,
                "attributes": {
                    "texture_consistency": 0.5 + 0.01 * i,
                    "edge_quality": 0.4 + 0.01 * i,
                    "color_distribution": 0.3 + 0.01 * i,
                    "geometric_coherence": 0.6 + 0.005 * i,
                    "semantic_plausibility": 0.7 - 0.01 * i,
                    "frequency_artifacts": 0.2 + 0.02 * i,
                },
            })

        with open(cls.ann_file, "w") as f:
            for entry in annotations:
                f.write(json.dumps(entry) + "\n")

        from data.xgenbench import XGenBench
        cls.bench = XGenBench(
            annotation_file=cls.ann_file,
            image_root=cls.tmpdir,
        )

    def test_evaluate_attributes_perfect(self):
        """Perfect predictions should yield MAE close to 0."""
        predictions = []
        for entry in self.bench.data:
            predictions.append({
                "image_path": entry["image_path"],
                "attributes": entry["attributes"],
            })

        results = self.bench.evaluate_attributes(predictions)
        self.assertIn("MAE_overall", results)
        self.assertAlmostEqual(results["MAE_overall"], 0.0, places=5)

    def test_evaluate_attributes_random(self):
        """Random attribute predictions should yield non-zero MAE."""
        rng = np.random.RandomState(42)
        predictions = []
        attr_names = [
            "texture_consistency", "edge_quality", "color_distribution",
            "geometric_coherence", "semantic_plausibility", "frequency_artifacts",
        ]
        for entry in self.bench.data:
            predictions.append({
                "image_path": entry["image_path"],
                "attributes": {name: float(rng.uniform(0, 1)) for name in attr_names},
            })

        results = self.bench.evaluate_attributes(predictions)
        self.assertIn("MAE_overall", results)
        self.assertGreater(results["MAE_overall"], 0.0)

    def test_evaluate_attributes_spearman(self):
        """With enough data, Spearman correlation should be computed."""
        # Use perfect predictions -- Spearman should be 1.0
        predictions = []
        for entry in self.bench.data:
            predictions.append({
                "image_path": entry["image_path"],
                "attributes": entry["attributes"],
            })

        results = self.bench.evaluate_attributes(predictions)
        # texture_consistency varies across samples, so Spearman should be 1.0
        self.assertIn("Spearman_texture_consistency", results)
        self.assertAlmostEqual(results["Spearman_texture_consistency"], 1.0, places=3)

    def test_evaluate_attributes_empty(self):
        """Empty predictions should return empty dict."""
        results = self.bench.evaluate_attributes([])
        self.assertNotIn("MAE_overall", results)

    def test_evaluate_attributes_missing_image(self):
        """Predictions with unknown image_path should be ignored."""
        predictions = [
            {"image_path": "nonexistent.png",
             "attributes": {"texture_consistency": 0.5}},
        ]
        results = self.bench.evaluate_attributes(predictions)
        self.assertNotIn("MAE_overall", results)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
