# AGENTS.md — Driving GenStack with an LLM agent

This file is a contract between **you** (a Claude / Codex / Cursor / Gemini / Aider / similar
LLM agent) and the GenStack codebase. Read it before issuing any non-trivial command in
this repo.

The intent: an agent should be able to land a useful change (reproduce a table, run a
sweep, train a sub-component, write a sibling script) **without re-deriving the design from
the paper every time.**

---

## 1. Repo mental model

```
GenStack/
├── pact/                  # discriminative branch (CLIP frozen + 8 FPT + 128-prototype PGAD + classifier)
├── prism_sft_ablation/    # InternVL3-8B reasoner LoRA recipe + view-masking ablation
├── reproduce/             # cached (p_v, b_p, CLIP-CLS, label) tuples — NO images, NO weights
├── paper/                 # IJCB 2026 LaTeX (main + supplementary)
└── AGENTS.md              # this file
```

The MoE meta-learner glue lives under `pact/scripts/` (not its own folder) because it
shares the cached-predictions data path with everything else.

**Three branches, three time scales:**
| Branch | Trains in | Inference cost | What you should *not* re-train |
|---|---|---|---|
| MoE meta-learner | <1 minute (CPU) | <0.1 ms/img | rarely worth re-training; sweeps are cheap |
| Pact | ~6 h on 1×A100 | ~10 ms/img | unless you change the backbone or PGAD layout |
| Prism-SFT | ~12 h on 4×A100 80GB | ~1 s/img | almost never; use cached predictions |

If you can solve the task with `reproduce/cached_predictions/`, do that first.

---

## 2. Hard rules

These are non-negotiable. Violations break reproducibility or burn GPU budget.

1. **Do not re-train Prism-SFT casually.** It costs 4×A100 × 12 h ≈ ~$200 of compute.
   Always check whether the cached `b_p` predictions answer the question first.
2. **Do not commit checkpoint binaries.** `.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`
   are gitignored. Push to Hugging Face Hub and link from `README.md` instead.
3. **Do not commit HydraFake images.** We do not own the data; redistribution violates
   the original license. Only derived per-image scores live in `reproduce/`.
4. **Do not change the random seeds in `reproduce_*.py` scripts** unless you also update
   the paper. Default seed is `42` everywhere; cross-seed runs use `{42, 0, 7}`.
5. **Do not "improve" the MoE by adding hidden layers.** The expert is a 50-tree
   `GradientBoostingClassifier(max_depth=2)` for a reason — it's the smallest model
   that captures the non-monotone fusion rule. Bigger models overfit. See
   `paper/supplementary.tex` §B (Table S2).
6. **Do not edit `paper/cvpr.sty`.** It's the IJCB-frozen review-mode style file with
   the "IJCB 2026" badge hard-coded at line 624.

---

## 3. Common tasks — agent playbook

### 3.1 "Reproduce the headline number"
```bash
python pact/scripts/reproduce_genstack_k15.py --cache reproduce/cached_predictions
```
Should print `Avg=92.83` ± noise from float ordering. If it doesn't, **stop** and check:
1. Did `reproduce/cached_predictions/c1e_merged_paths.json` deserialize fully? (52,266 rows)
2. Are the 8 NPY chunks under `c1e_clip_chunks/` all present and uncorrupted?
3. Is `sklearn` ≥ 1.3 installed? Older versions change `GradientBoostingClassifier` defaults.

### 3.2 "Add a new ablation"
The cheapest place to live is `pact/scripts/`. Pattern:
1. Load cached preds the same way every other script does (`load_cached_preds()` helper).
2. Build the gate / experts inside a `5-fold StratifiedKFold(random_state=42)` loop.
3. Save results to `reproduce/cached_predictions/<your_ablation>.json`.
4. Add a row to `paper/supplementary.tex` and mention it in this `AGENTS.md` if the
   pattern would be useful for future agents.

### 3.3 "Investigate why Pact is wrong on generator X"
Don't retrain. Run `pact/evaluation/per_generator_breakdown.py` against cached `p_v`. If
`p_v` is genuinely random (~0.5 mean), Pact is the bottleneck and the paper section to
update is §4.4. If `p_v` is decisive but the MoE softens it, the gate is the bottleneck —
update §4.3.

### 3.4 "Add a new view to Prism-SFT"
Files to touch:
- `prism_sft_ablation/multiview/extract_<view>.py` — view extraction
- `prism_sft_ablation/multiview/build_sft_jsonl.py` — multi-image record format
- `prism_sft_ablation/multiview/train_full.sh` — pass new `--views` arg
Then evaluate with `prism_sft_ablation/multiview/run_inference_views.sh` *before*
declaring victory. The view-masking baseline in Table S6 is a reasonable diff target.

### 3.5 "Update the paper"
- `paper/main.tex` — main paper (8 pages + refs)
- `paper/supplementary.tex` — supplementary (sections A–O)
- Both share the same custom commands: `\method`, `\vfive` (= Pact), `\prism` (= Prism-SFT).
- Use `\textsc{...}` for method names. Don't introduce new acronyms without checking they
  don't already exist.

---

## 4. Verification before claiming success

Before you tell the user "done", verify:

| Claim | How to verify | What to paste back |
|---|---|---|
| "Reproduced Avg=92.83" | run `reproduce_genstack_k15.py` | the script's last 2 lines |
| "Added new ablation" | run the new script + a `git diff --stat` | numbers + diff summary |
| "Updated paper" | `cd paper && pdflatex main.tex` runs clean | page count + any warnings |
| "Trained new model" | the eval script's per-split numbers | full breakdown, not just Avg |

Do **not** claim success based on "the script started running" or "no exception was
raised." A subtle silent breakage (e.g., gate trained on the wrong target) will look
fine until someone reads the table.

---

## 5. Style notes

- Python: 3.10+, type hints where helpful, no aggressive refactoring of existing code.
- Bash: stick to the Conda env names already used (`Qwen2.5` for Prism-SFT, `genstack`
  for the rest).
- LaTeX: keep both main and supplementary using `cvpr.sty` review mode. The supplementary
  is `S\arabic{table}` / `S\arabic{figure}` / `\Alph{section}` so refs don't collide.
- Logs: prefer `print()` to `logging.*` in scripts (tooling around this repo greps the stdout).
- Don't add docstrings longer than one line. Don't add comments explaining what
  well-named code does. (Both rules apply to *added* code; existing comments stay.)

---

## 6. When in doubt

1. Check `paper/main.tex` and `paper/supplementary.tex` — they are the spec of what this
   code is supposed to compute.
2. Check `reproduce/cached_predictions/` — if the answer is in here, you don't need GPUs.
3. If you would change a number that already appears in the paper, **stop and ask the
   human**.
