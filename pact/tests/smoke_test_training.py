#!/usr/bin/env python3
"""
XGenDet Stage 1 Smoke Test.

Verifies the complete Stage 1 training pipeline end-to-end on CPU
using synthetic random data. Checks:
  1. Model build on CPU
  2. Synthetic dataset + DataLoader
  3. Forward pass produces correct output shapes
  4. Loss computation (XGenDetLoss)
  5. Backward pass + optimizer step
  6. Loss decreases over 5 training steps
  7. Gradients exist on trainable parameters
  8. Heatmap values in valid range
  9. Confidence values in [0, 1]
  10. Family logits have 4 classes
  11. Attribute scores have 6 dimensions
  12. Inference via model.detect()
  13. Checkpoint save/load roundtrip
"""

import os
import sys
import time
import tempfile
import traceback

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.xgendet import XGenDet
from training.losses import XGenDetLoss

# ---- Configuration ----
BATCH_SIZE = 4
NUM_REAL = 50
NUM_FAKE = 50
NUM_STEPS = 5
CROP_SIZE = 224
NUM_FAMILIES = 4      # Real=0, GAN=1, Diffusion=2, AR=3
NUM_ATTRIBUTES = 6
NUM_PROTOTYPES = 128
DEVICE = torch.device("cpu")

# Use smaller model settings for faster smoke test
MODEL_KWARGS = dict(
    clip_model_name="ViT-L/14",
    num_prompt_tokens=8,
    num_prototypes=NUM_PROTOTYPES,
    proto_dim=128,
    shuffle_patch_size=32,
)


# ---- Synthetic Dataset ----
class SyntheticRealFakeDataset(Dataset):
    """Synthetic dataset with random tensors mimicking real/fake images."""

    def __init__(self, num_real: int, num_fake: int, crop_size: int = 224):
        super().__init__()
        self.crop_size = crop_size
        self.data = []
        # Real images: label=0, family=0
        for _ in range(num_real):
            self.data.append((0, 0))
        # Fake images: label=1, family randomly in {1, 2, 3}
        for _ in range(num_fake):
            family = torch.randint(1, NUM_FAMILIES, (1,)).item()
            self.data.append((1, family))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, family = self.data[idx]
        # Random tensor normalised like CLIP input
        img = torch.randn(3, self.crop_size, self.crop_size)
        return img, label, family


# ---- Test Utilities ----
class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def record(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed.append(name)
            print(f"  [PASS] {name}")
        else:
            self.failed.append((name, detail))
            print(f"  [FAIL] {name}: {detail}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 60)
        print(f"SMOKE TEST SUMMARY: {len(self.passed)}/{total} passed")
        print("=" * 60)
        if self.failed:
            print("\nFailed tests:")
            for name, detail in self.failed:
                print(f"  - {name}: {detail}")
            print("\nRESULT: FAIL")
            return False
        else:
            print("\nRESULT: PASS")
            return True


# ---- Main Smoke Test ----
def run_smoke_test():
    t0 = time.time()
    results = TestResult()

    # ------------------------------------------------------------------
    # 1. Build model on CPU
    # ------------------------------------------------------------------
    print("\n[1] Building XGenDet model on CPU ...")
    try:
        model = XGenDet(**MODEL_KWARGS).to(DEVICE)
        model.train()
        param_counts = model.count_trainable_params()
        print(f"     Trainable params: {param_counts['total']:,}")
        results.record("model_build", True)
    except Exception as e:
        results.record("model_build", False, str(e))
        traceback.print_exc()
        results.summary()
        return

    # ------------------------------------------------------------------
    # 2. Create synthetic dataset + DataLoader
    # ------------------------------------------------------------------
    print("\n[2] Creating synthetic dataset ...")
    try:
        dataset = SyntheticRealFakeDataset(NUM_REAL, NUM_FAKE, CROP_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        results.record("dataset_creation", True)
    except Exception as e:
        results.record("dataset_creation", False, str(e))
        traceback.print_exc()
        results.summary()
        return

    # ------------------------------------------------------------------
    # 3. Forward pass - check output shapes
    # ------------------------------------------------------------------
    print("\n[3] Running forward pass ...")
    try:
        imgs, labels, families = next(iter(loader))
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        families = families.to(DEVICE)

        outputs = model(imgs, return_heatmap=True)

        B = imgs.shape[0]
        expected_shapes = {
            "binary_logit": (B, 1),
            "confidence": (B, 1),
            "family_logit": (B, NUM_FAMILIES),
            "heatmap": (B, 1, CROP_SIZE, CROP_SIZE),
            "proto_activations": (B, NUM_PROTOTYPES),
            "proto_spatial_maps": (B, NUM_PROTOTYPES, 16, 16),
            "attr_scores": (B, NUM_ATTRIBUTES),
            "proto_features": (B, NUM_PROTOTYPES, 128),
            "patches_orig": (B, 256, 1024),  # 16x16 patches, hidden_dim=1024
        }

        all_shapes_ok = True
        for key, expected in expected_shapes.items():
            actual = outputs[key].shape
            if actual != expected:
                results.record(f"shape_{key}", False,
                               f"expected {expected}, got {actual}")
                all_shapes_ok = False
            else:
                results.record(f"shape_{key}", True)

        results.record("forward_pass", True)
    except Exception as e:
        results.record("forward_pass", False, str(e))
        traceback.print_exc()
        results.summary()
        return

    # ------------------------------------------------------------------
    # 4. Value range checks
    # ------------------------------------------------------------------
    print("\n[4] Checking value ranges ...")
    heatmap = outputs["heatmap"]
    confidence = outputs["confidence"]
    family_logit = outputs["family_logit"]
    attr_scores = outputs["attr_scores"]

    hm_min = heatmap.min().item()
    hm_max = heatmap.max().item()
    results.record("heatmap_range",
                   0.0 <= hm_min and hm_max <= 1.0,
                   f"min={hm_min:.4f}, max={hm_max:.4f}")

    conf_min = confidence.min().item()
    conf_max = confidence.max().item()
    results.record("confidence_range",
                   0.0 <= conf_min and conf_max <= 1.0,
                   f"min={conf_min:.4f}, max={conf_max:.4f}")

    results.record("family_logit_classes",
                   family_logit.shape[-1] == NUM_FAMILIES,
                   f"expected {NUM_FAMILIES}, got {family_logit.shape[-1]}")

    results.record("attr_scores_dims",
                   attr_scores.shape[-1] == NUM_ATTRIBUTES,
                   f"expected {NUM_ATTRIBUTES}, got {attr_scores.shape[-1]}")

    # ------------------------------------------------------------------
    # 5. Loss computation
    # ------------------------------------------------------------------
    print("\n[5] Computing loss ...")
    try:
        criterion = XGenDetLoss(
            w_family=0.5,
            w_proto_div=0.3,
            w_proto_compact=0.2,
            w_heatmap=0.3,
            w_attr=0.1,
            w_calib=0.2,
        )

        losses = criterion(
            outputs, labels, families,
            prototype_module=model.prototype_module,
        )

        expected_loss_keys = ["total", "cls", "family", "proto_div",
                              "proto_compact", "heatmap", "attr", "calib"]
        for k in expected_loss_keys:
            results.record(f"loss_key_{k}",
                           k in losses and torch.isfinite(losses[k]),
                           f"missing or non-finite" if k not in losses else "")

        results.record("loss_computation", True)
    except Exception as e:
        results.record("loss_computation", False, str(e))
        traceback.print_exc()
        results.summary()
        return

    # ------------------------------------------------------------------
    # 6. Backward + optimizer step over 5 iterations; verify loss decrease
    # ------------------------------------------------------------------
    print("\n[6] Training loop ({} steps) ...".format(NUM_STEPS))
    try:
        param_groups = model.get_trainable_params()
        optimizer_params = []
        for group in param_groups:
            lr_scale = group.get("lr_scale", 1.0)
            optimizer_params.append({
                "params": group["params"],
                "lr": 1e-4 * lr_scale,
                "name": group.get("name", "unnamed"),
            })
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)

        loss_values = []
        data_iter = iter(loader)
        for step in range(NUM_STEPS):
            try:
                imgs, labels, families = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                imgs, labels, families = next(data_iter)

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            families = families.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs, return_heatmap=True)
            losses = criterion(
                outputs, labels, families,
                prototype_module=model.prototype_module,
            )
            total_loss = losses["total"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lv = total_loss.item()
            loss_values.append(lv)
            print(f"     step {step+1}/{NUM_STEPS}  loss={lv:.4f}")

        results.record("backward_and_step", True)

        # Check loss decrease: final < initial (allowing some tolerance)
        decreased = loss_values[-1] < loss_values[0]
        results.record("loss_decreases",
                       decreased,
                       f"initial={loss_values[0]:.4f}, final={loss_values[-1]:.4f}")

    except Exception as e:
        results.record("backward_and_step", False, str(e))
        traceback.print_exc()
        results.summary()
        return

    # ------------------------------------------------------------------
    # 7. Verify gradients exist on trainable parameters
    # ------------------------------------------------------------------
    print("\n[7] Checking gradients ...")
    has_grad = False
    total_trainable = 0
    grads_present = 0
    for p in model.parameters():
        if p.requires_grad:
            total_trainable += 1
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                grads_present += 1
                has_grad = True

    results.record("gradients_exist",
                   has_grad,
                   f"{grads_present}/{total_trainable} params have nonzero grad")

    # ------------------------------------------------------------------
    # 8. Inference via model.detect()
    # ------------------------------------------------------------------
    print("\n[8] Testing inference (model.detect) ...")
    try:
        test_imgs = torch.randn(2, 3, CROP_SIZE, CROP_SIZE, device=DEVICE)
        with torch.no_grad():
            det = model.detect(test_imgs)

        results.record("detect_prediction",
                       det["prediction"].shape == (2,),
                       f"shape={det['prediction'].shape}")
        results.record("detect_confidence",
                       det["confidence"].shape == (2,),
                       f"shape={det['confidence'].shape}")
        results.record("detect_family",
                       det["family"].shape == (2,),
                       f"shape={det['family'].shape}")
        results.record("detect_heatmap",
                       det["heatmap"].shape == (2, 1, CROP_SIZE, CROP_SIZE),
                       f"shape={det['heatmap'].shape}")
        results.record("detect_attr_scores",
                       det["attr_scores"].shape == (2, NUM_ATTRIBUTES),
                       f"shape={det['attr_scores'].shape}")
        results.record("detect_proto_activations",
                       det["proto_activations"].shape == (2, NUM_PROTOTYPES),
                       f"shape={det['proto_activations'].shape}")

        # Confidence must still be in [0, 1]
        c = det["confidence"]
        results.record("detect_confidence_range",
                       c.min().item() >= 0.0 and c.max().item() <= 1.0,
                       f"min={c.min().item():.4f}, max={c.max().item():.4f}")

        results.record("inference", True)
    except Exception as e:
        results.record("inference", False, str(e))
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 9. Checkpoint save / load roundtrip
    # ------------------------------------------------------------------
    print("\n[9] Testing checkpoint save/load roundtrip ...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "smoke_test.pth")

            # Save
            torch.save({
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

            # Verify file exists and is non-empty
            file_size = os.path.getsize(ckpt_path)
            results.record("checkpoint_saved",
                           file_size > 0,
                           f"file size={file_size} bytes")

            # Load into fresh model
            model2 = XGenDet(**MODEL_KWARGS).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model2.load_state_dict(ckpt["model_state_dict"])
            results.record("checkpoint_loaded", True)

            # Verify parameters match
            all_match = True
            for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                            model2.named_parameters()):
                if n1 != n2 or not torch.equal(p1.data, p2.data):
                    all_match = False
                    break
            results.record("checkpoint_params_match", all_match,
                           "" if all_match else f"mismatch at {n1}")

            # Verify loaded model produces same outputs.
            # Note: shuffle_patches uses torch.randperm, so we must seed
            # before each forward call to get identical shuffled inputs.
            model.eval()
            model2.eval()
            with torch.no_grad():
                torch.manual_seed(9999)
                out1 = model(test_imgs, return_heatmap=True)
                torch.manual_seed(9999)
                out2 = model2(test_imgs, return_heatmap=True)

            # Confidence should be identical (deterministic with same seed)
            match = torch.allclose(out1["confidence"], out2["confidence"], atol=1e-5)
            results.record("checkpoint_output_match", match,
                           "" if match else "outputs differ after load")

    except Exception as e:
        results.record("checkpoint_roundtrip", False, str(e))
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\nElapsed time: {elapsed:.1f}s")
    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run_smoke_test()
