"""
Reward functions for Part-Aware GRPO (Stage 3).

Implements the VIGIL-inspired composite reward:
  R = R_acc + λ₁·R_part + λ₂·R_cons + λ₃·R_fmt

Each component:
  R_acc:  Accuracy reward — did the model get real/fake correct?
  R_part: Part-aware reward — does reasoning mention correct suspicious regions?
  R_cons: Consistency reward — is reasoning consistent with the conclusion?
  R_fmt:  Format compliance — proper <think>...<answer> structure?
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>real/fake</answer> tags."""
    match = re.search(r'<answer>\s*(real|fake)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: look for standalone real/fake at end
    text_lower = text.lower().strip()
    if text_lower.endswith("fake"):
        return "fake"
    if text_lower.endswith("real"):
        return "real"
    return None


def extract_thinking(text: str) -> str:
    """Extract content from <think>...</think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_mentioned_regions(text: str) -> list:
    """Extract facial regions mentioned in the reasoning."""
    text_lower = text.lower()
    regions = []
    region_keywords = {
        "eye": ["eye", "eyes", "eyelid", "eyelash", "pupil", "iris", "gaze"],
        "nose": ["nose", "nostril", "nasal"],
        "mouth": ["mouth", "lip", "lips", "teeth", "smile"],
        "forehead": ["forehead", "hairline"],
        "cheek": ["cheek", "cheeks"],
        "jaw": ["jaw", "chin", "jawline"],
        "ear": ["ear", "ears"],
        "skin": ["skin", "texture", "pore", "pores", "complexion"],
        "boundary": ["boundary", "border", "edge", "seam", "blending"],
        "hair": ["hair", "hairline"],
    }
    for region, keywords in region_keywords.items():
        if any(kw in text_lower for kw in keywords):
            regions.append(region)
    return regions


# ── Individual Reward Components ─────────────────────────────────────────────

def reward_accuracy(response: str, ground_truth: str) -> float:
    """
    R_acc: +1.0 for correct answer, -1.0 for wrong.
    If no answer can be extracted, -0.5 penalty.
    """
    predicted = extract_answer(response)
    if predicted is None:
        return -0.5
    return 1.0 if predicted == ground_truth.lower() else -1.0


def reward_part_aware(
    response: str,
    ground_truth: str,
    srm_evidence: Optional[dict] = None,
) -> float:
    """
    R_part: Does the reasoning mention relevant facial regions?

    For fake images: reward mentioning high-noise regions (verified by SRM).
    For real images: reward NOT flagging specific parts as suspicious.
    """
    thinking = extract_thinking(response)
    if not thinking:
        return -0.2  # No reasoning at all

    mentioned = extract_mentioned_regions(thinking)

    if ground_truth.lower() == "fake":
        # Fake: should mention suspicious regions
        if not mentioned:
            return -0.3  # No regions mentioned for a fake image

        # If SRM evidence available, check if mentioned regions align with high-noise areas
        if srm_evidence:
            high_noise_regions = srm_evidence.get("high_noise_regions", [])
            if high_noise_regions:
                overlap = set(mentioned) & set(high_noise_regions)
                if overlap:
                    return 0.5  # Correctly identified suspicious regions
                return 0.0  # Mentioned regions but not the right ones

        return 0.3  # Mentioned regions (no SRM to verify)

    else:
        # Real: should NOT flag specific parts as highly suspicious
        suspicious_language = any(
            word in thinking.lower()
            for word in ["suspicious", "anomaly", "artifact", "manipulation",
                         "inconsisten", "distort", "unnatural"]
        )
        if mentioned and suspicious_language:
            return -0.2  # Incorrectly flagging a real image
        return 0.3  # Correctly not flagging


def reward_consistency(response: str) -> float:
    """
    R_cons: Is the reasoning internally consistent with the conclusion?

    Detects contradictions like reasoning says "appears natural" but concludes "fake",
    or reasoning says "clear artifacts" but concludes "real".
    """
    thinking = extract_thinking(response)
    answer = extract_answer(response)

    if not thinking or not answer:
        return 0.0  # Can't evaluate

    thinking_lower = thinking.lower()

    # Indicators of fake
    fake_indicators = [
        "artifact", "manipulation", "forgery", "synthetic", "generated",
        "inconsisten", "anomal", "unnatural", "suspicious", "distort",
        "blurr", "seam", "boundary issue", "noise anomaly",
    ]
    # Indicators of real
    real_indicators = [
        "authentic", "genuine", "natural", "consistent", "real photograph",
        "no sign", "no evidence", "appears real", "appears natural",
        "no artifact", "uniform noise",
    ]

    fake_score = sum(1 for ind in fake_indicators if ind in thinking_lower)
    real_score = sum(1 for ind in real_indicators if ind in thinking_lower)

    if answer == "fake":
        if fake_score > real_score:
            return 0.3  # Consistent: reasoning supports fake conclusion
        elif real_score > fake_score + 1:
            return -0.5  # Contradictory: reasoning says real but concludes fake
        return 0.0

    else:  # answer == "real"
        if real_score > fake_score:
            return 0.3  # Consistent: reasoning supports real conclusion
        elif fake_score > real_score + 1:
            return -0.5  # Contradictory: reasoning says fake but concludes real
        return 0.0


def reward_format(response: str) -> float:
    """
    R_fmt: Does the response follow the expected format?
    Expected: <think>...</think> followed by <answer>real/fake</answer>
    """
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>\s*(real|fake)\s*</answer>', response, re.IGNORECASE))

    if has_think and has_answer:
        return 0.2
    elif has_answer:
        return 0.0  # Answer present but no thinking
    else:
        return -0.3  # Missing answer tag


# ── Composite Reward ─────────────────────────────────────────────────────────

def compute_composite_reward(
    response: str,
    ground_truth: str,
    srm_evidence: Optional[dict] = None,
    lambda_part: float = 0.5,
    lambda_cons: float = 0.3,
    lambda_fmt: float = 0.2,
) -> dict:
    """
    Compute the full VIGIL-inspired composite reward.

    R = R_acc + λ₁·R_part + λ₂·R_cons + λ₃·R_fmt

    Returns dict with individual components and total.
    """
    r_acc = reward_accuracy(response, ground_truth)
    r_part = reward_part_aware(response, ground_truth, srm_evidence)
    r_cons = reward_consistency(response)
    r_fmt = reward_format(response)

    total = r_acc + lambda_part * r_part + lambda_cons * r_cons + lambda_fmt * r_fmt

    return {
        "total": total,
        "r_acc": r_acc,
        "r_part": r_part,
        "r_cons": r_cons,
        "r_fmt": r_fmt,
    }


# ── Batch Reward Function for trl GRPOTrainer ────────────────────────────────

def grpo_reward_fn(
    completions: list[str],
    ground_truths: list[str],
    srm_evidences: list[Optional[dict]] = None,
    **kwargs,
) -> list[float]:
    """
    Reward function compatible with trl GRPOTrainer.

    Args:
        completions: List of generated responses
        ground_truths: List of ground truth labels ("real" or "fake")
        srm_evidences: Optional list of SRM evidence dicts

    Returns:
        List of float rewards
    """
    rewards = []
    for i, completion in enumerate(completions):
        gt = ground_truths[i] if i < len(ground_truths) else "unknown"
        srm = srm_evidences[i] if srm_evidences and i < len(srm_evidences) else None
        result = compute_composite_reward(completion, gt, srm)
        rewards.append(result["total"])
    return rewards
