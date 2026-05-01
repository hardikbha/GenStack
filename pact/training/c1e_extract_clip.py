"""C1-E worker — extract frozen CLIP ViT-L/14 CLS embeddings on a slice
[--start, --end) of the merged path list, write to .npy chunks."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import clip  # OpenAI clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument(
        '--out_dir',
        default=
        '/home/sachin.chaudhary/xgendet/checkpoints/final_results/c1e_clip_chunks'
    )
    ap.add_argument(
        '--paths_json',
        default=
        '/home/sachin.chaudhary/xgendet/checkpoints/final_results/c1e_merged_paths.json'
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_npy = f"{args.out_dir}/feat_{args.start:06d}_{args.end:06d}.npy"
    out_idx = f"{args.out_dir}/feat_{args.start:06d}_{args.end:06d}.json"
    if os.path.exists(out_npy) and os.path.exists(out_idx):
        print(f"[skip] {out_npy} exists", flush=True)
        return

    print(f"[init] device={args.device} slice=[{args.start},{args.end})",
          flush=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, preprocess = clip.load('ViT-L/14', device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[init] CLIP ViT-L/14 loaded on {device}", flush=True)

    merged = json.load(open(args.paths_json))[args.start:args.end]
    print(f"[init] {len(merged)} samples to process", flush=True)

    feats, kept_idx = [], []
    bad = 0
    t0 = time.time()
    BSZ = args.batch
    buf_imgs, buf_pos = [], []

    def flush():
        nonlocal buf_imgs, buf_pos
        if not buf_imgs: return
        x = torch.stack(buf_imgs).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                f = model.encode_image(x)
        f = f.float().cpu().numpy()
        feats.append(f)
        kept_idx.extend(buf_pos)
        buf_imgs.clear()
        buf_pos.clear()

    for i, r in enumerate(merged):
        try:
            img = Image.open(r['path']).convert('RGB')
            buf_imgs.append(preprocess(img))
            buf_pos.append(i)
        except Exception:
            bad += 1
        if len(buf_imgs) >= BSZ:
            flush()
            if (i + 1) % (BSZ * 20) == 0:
                rate = (i + 1) / max(1, time.time() - t0)
                eta = (len(merged) - i - 1) / max(1, rate)
                print(
                    f"[progress] {i+1}/{len(merged)}  bad={bad}  "
                    f"{rate:.1f} img/s  eta {eta/60:.1f} min",
                    flush=True)
    flush()

    feats = np.concatenate(feats, axis=0) if feats else np.zeros(
        (0, 768), np.float32)
    np.save(out_npy, feats.astype(np.float32))
    json.dump(
        {
            'kept_idx_in_slice': kept_idx,
            'slice_start': args.start,
            'slice_end': args.end,
            'bad': bad
        }, open(out_idx, 'w'))
    print(
        f"[done] {feats.shape} → {out_npy}  bad={bad}  "
        f"{(time.time()-t0)/60:.1f} min",
        flush=True)


if __name__ == '__main__':
    main()
