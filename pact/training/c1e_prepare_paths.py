"""C1-E prep — save merged (path, gen, split, label, p_v, b_p) for the 52K test
samples so each GPU worker can slice the same list deterministically."""
import json, os, re
from glob import glob
from pathlib import Path

v5_probs = json.load(
    open(
        '/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'
    ))
_ANS = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

prism = {}
for d in ([
        f'/home/sachin.chaudhary/veritas_clone/result_prism_worker{i}'
        for i in range(4)
] + [
        f'/home/sachin.chaudhary/veritas_clone/result_prism_helper{i}'
        for i in range(13)
]):
    if not os.path.isdir(d): continue
    for f in glob(f"{d}/prism_*.jsonl"):
        bn = Path(f).name
        if bn.startswith(('prism_v2_', 'prism_mipo_')): continue
        name = bn.replace('.jsonl', '').split('-')[0].replace('prism_', '', 1)
        gen = name.replace('_part0', '').replace('_part1', '')
        with open(f) as fh:
            for line in fh:
                if not line.strip(): continue
                try:
                    d_ = json.loads(line)
                except Exception:
                    continue
                imgs = d_.get('images', [])
                if not imgs or d_.get('label') is None: continue
                p = imgs[0].get('path') if isinstance(imgs[0],
                                                      dict) else imgs[0]
                if not p: continue
                m = _ANS.search(d_.get('response', ''))
                if not m: continue
                prism[p] = {
                    'pred': 1 if m.group(1).lower() == 'fake' else 0,
                    'label': int(d_['label']),
                    'generator': gen,
                    'split': gen.split('_')[0]
                }

merged = []
for p, d in prism.items():
    vp = v5_probs.get(p)
    if vp is None: continue
    merged.append({
        'path': p,
        'v5': vp,
        'prism': d['pred'],
        'label': d['label'],
        'generator': d['generator'],
        'split': d['split']
    })

# stable order
merged.sort(key=lambda r: r['path'])
out = '/home/sachin.chaudhary/xgendet/checkpoints/final_results/c1e_merged_paths.json'
json.dump(merged, open(out, 'w'))
print(f"saved {len(merged)} entries → {out}")
