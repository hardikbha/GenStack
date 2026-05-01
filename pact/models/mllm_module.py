"""
XGenDet Stage 2: MLLM Explainability Module.

Wraps Qwen2.5-VL-7B with LoRA for generating attribute scores
and natural language explanations from image + heatmap overlays.

Input: Original image + heatmap overlay + Stage 1 structured output
Output: 6 attribute scores + NL explanation text
"""

import re
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np


# Prompt template for forensic analysis
FORENSIC_PROMPT = """You are a forensic image analyst specializing in AI-generated image detection. Analyze this image for signs of AI generation.

Stage 1 detector results:
- Prediction: {prediction} (confidence: {confidence:.1%})
- Generator family: {family}
- Top attribute activations: {attr_summary}

The second image is a heatmap overlay highlighting regions the detector found suspicious (red = more suspicious).

Provide your analysis in this exact format:

<attributes>
texture_consistency: [score 0.0-1.0]
edge_quality: [score 0.0-1.0]
color_distribution: [score 0.0-1.0]
geometric_coherence: [score 0.0-1.0]
semantic_plausibility: [score 0.0-1.0]
frequency_artifacts: [score 0.0-1.0]
</attributes>
<explanation>
[2-4 sentence explanation of why this image appears real or AI-generated, referencing specific visual evidence and regions highlighted in the heatmap]
</explanation>"""

ATTRIBUTE_NAMES = [
    "texture_consistency",
    "edge_quality",
    "color_distribution",
    "geometric_coherence",
    "semantic_plausibility",
    "frequency_artifacts",
]

FAMILY_NAMES = ["Real", "GAN", "Diffusion", "Autoregressive"]


def parse_mllm_output(text: str) -> Dict[str, Any]:
    """Parse structured output from MLLM response."""
    result = {"attributes": {}, "explanation": ""}

    # Parse attributes
    attr_match = re.search(r"<attributes>(.*?)</attributes>", text, re.DOTALL)
    if attr_match:
        attr_text = attr_match.group(1)
        for name in ATTRIBUTE_NAMES:
            score_match = re.search(rf"{name}:\s*([\d.]+)", attr_text)
            if score_match:
                result["attributes"][name] = float(score_match.group(1))
            else:
                result["attributes"][name] = 0.5  # default

    # Parse explanation
    expl_match = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    if expl_match:
        result["explanation"] = expl_match.group(1).strip()
    else:
        # Fallback: use everything after attributes block
        if attr_match:
            remainder = text[attr_match.end():].strip()
            if remainder:
                result["explanation"] = remainder
        else:
            result["explanation"] = text.strip()

    return result


class MLLMExplainer(nn.Module):
    """
    MLLM-based explainability module using Qwen2.5-VL-7B.

    Operates as a separate stage from the detection backbone.
    Takes the original image, heatmap overlay, and Stage 1 outputs
    to generate detailed attribute analysis and natural language explanation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: tuple = ("q_proj", "v_proj"),
        max_new_tokens: int = 512,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "target_modules": list(lora_target_modules),
        }

    def load_model(self):
        """Load model and processor. Called separately to avoid loading at init."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"Loading MLLM: {self.model_name}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if self.use_lora:
            self._apply_lora()

        # Freeze base model, only LoRA params are trainable
        if self.use_lora:
            for name, param in self.model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"MLLM loaded. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        try:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=self._lora_config["r"],
                lora_alpha=self._lora_config["alpha"],
                target_modules=self._lora_config["target_modules"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()
        except ImportError:
            print("WARNING: peft not installed. Running without LoRA.")
            self.use_lora = False

    def _build_prompt(
        self,
        stage1_outputs: Dict[str, Any],
    ) -> str:
        """Build the forensic analysis prompt from Stage 1 outputs."""
        confidence = stage1_outputs.get("confidence", 0.5)
        prediction = "FAKE" if confidence > 0.5 else "REAL"
        family_idx = stage1_outputs.get("family", 0)
        family = FAMILY_NAMES[family_idx] if family_idx < len(FAMILY_NAMES) else "Unknown"

        # Summarize attribute scores
        attr_scores = stage1_outputs.get("attr_scores", [0.5] * 6)
        attr_names_short = ["Texture", "Edges", "Color", "Geometry", "Semantics", "Frequency"]
        attr_summary = ", ".join(
            f"{n}={s:.2f}" for n, s in zip(attr_names_short, attr_scores)
        )

        return FORENSIC_PROMPT.format(
            prediction=prediction,
            confidence=confidence,
            family=family,
            attr_summary=attr_summary,
        )

    def _create_heatmap_overlay(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Create heatmap overlay on image for MLLM input."""
        import matplotlib.cm as cm

        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize and resize heatmap
        hmap = heatmap.squeeze()
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

        if hmap.shape != img_array.shape[:2]:
            hmap_pil = Image.fromarray((hmap * 255).astype(np.uint8))
            hmap_pil = hmap_pil.resize(
                (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
            )
            hmap = np.array(hmap_pil).astype(np.float32) / 255.0

        cmap = cm.get_cmap("jet")
        heatmap_colored = cmap(hmap)[:, :, :3]

        overlay = alpha * heatmap_colored + (1 - alpha) * img_array
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    @torch.no_grad()
    def explain(
        self,
        image: Image.Image,
        heatmap: Optional[np.ndarray] = None,
        stage1_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate explanation for an image.

        Args:
            image: PIL Image (original)
            heatmap: numpy array heatmap from Stage 1
            stage1_outputs: dict with confidence, family, attr_scores

        Returns:
            dict with attributes (6 scores) and explanation (text)
        """
        if self.model is None:
            self.load_model()

        if stage1_outputs is None:
            stage1_outputs = {"confidence": 0.5, "family": 0, "attr_scores": [0.5] * 6}

        prompt = self._build_prompt(stage1_outputs)

        # Build message content
        content = [{"type": "image", "image": image}]

        if heatmap is not None:
            overlay = self._create_heatmap_overlay(image, heatmap)
            content.append({"type": "image", "image": overlay})

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Process inputs
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        # Decode (strip input tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Parse structured output
        parsed = parse_mllm_output(response)
        parsed["raw_response"] = response

        return parsed

    def prepare_training_data(
        self,
        image: Image.Image,
        heatmap: Optional[np.ndarray],
        stage1_outputs: Dict[str, Any],
        target_attributes: Dict[str, float],
        target_explanation: str,
    ) -> Dict[str, Any]:
        """
        Prepare a single training example for Stage 2 fine-tuning.

        Returns dict with tokenized inputs and labels.
        """
        prompt = self._build_prompt(stage1_outputs)

        # Build target response
        attr_lines = "\n".join(
            f"{name}: {target_attributes.get(name, 0.5):.2f}"
            for name in ATTRIBUTE_NAMES
        )
        target_response = f"<attributes>\n{attr_lines}\n</attributes>\n<explanation>\n{target_explanation}\n</explanation>"

        # Build messages
        content = [{"type": "image", "image": image}]
        if heatmap is not None:
            overlay = self._create_heatmap_overlay(image, heatmap)
            content.append({"type": "image", "image": overlay})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": target_response},
        ]

        return {
            "messages": messages,
            "target": target_response,
        }

    def save_lora_weights(self, save_path: str):
        """Save only LoRA adapter weights."""
        if self.use_lora and self.model is not None:
            self.model.save_pretrained(save_path)
            print(f"LoRA weights saved to {save_path}")

    def load_lora_weights(self, load_path: str):
        """Load LoRA adapter weights."""
        if self.model is None:
            self.load_model()
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, load_path)
            print(f"LoRA weights loaded from {load_path}")
