"""
Ensemble: v5 (ViT-L/14 @ 224px) + v14 (ViT-L/14@336px)
Each model runs its own TTA. Confidences averaged with optional weights.
"""
import os, sys, json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeTestDataset

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

MODELS = [
    {"ckpt": "checkpoints/v5_resume_hl/best_model.pth", "clip": "ViT-L/14",       "crop": 224, "w": 0.5},
    {"ckpt": "checkpoints/v14_336px/best_model.pth",    "clip": "ViT-L/14@336px",  "crop": 336, "w": 0.5},
]

TEST_DIR   = "/home/sachin.chaudhary/hydrafake/jsons/test"
DATA_ROOT  = "/home/sachin.chaudhary"
IMAGE_ROOT = "/home/sachin.chaudhary/hydrafake/test"
OUTPUT_DIR = "checkpoints/ensemble_v5_v14"
BATCH_SIZE = 64
NUM_WORKERS= 4

def tta_views(crop):
    native = int(crop * 256/224)
    return [
        T.Compose([T.Resize((crop,crop)), T.ToTensor(), T.Normalize(CLIP_MEAN,CLIP_STD)]),
        T.Compose([T.Resize((crop,crop)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(CLIP_MEAN,CLIP_STD)]),
        T.Compose([T.Resize((native,native)), T.CenterCrop(crop), T.ToTensor(), T.Normalize(CLIP_MEAN,CLIP_STD)]),
    ]

def load_model(spec, device):
    m = XGenDet(clip_model_name=spec["clip"])
    ckpt = torch.load(spec["ckpt"], map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k):v for k,v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval().to(device)

@torch.no_grad()
def predict(model, ds, views, device):
    all_p = []
    for view in views:
        ds.transform = view
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
        p, l = [], []
        for imgs, lab in loader:
            out = model(imgs.to(device), return_heatmap=False)
            p.extend(out["confidence"].squeeze(-1).cpu().tolist())
            l.extend(lab.tolist())
        all_p.append(np.array(p))
    return np.mean(all_p, axis=0), np.array(l)

def youden_thresh(p, l):
    if len(np.unique(l)) < 2: return 0.5
    fpr, tpr, th = roc_curve(l, p)
    return float(th[np.argmax(tpr-fpr)])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("Ensemble: v5 (224px, w=0.5) + v14 @336px (w=0.5) with TTA")
    print("="*70)

    loaded = [(load_model(s, device), tta_views(s["crop"]), s["w"]) for s in MODELS]
    print(f"Loaded {len(loaded)} models\n")

    results = {}
    for split in ["id","cm","cf","cd"]:
        sdir = os.path.join(TEST_DIR, split)
        if not os.path.isdir(sdir): continue
        results[split] = {"per_generator":{}}
        all_p, all_l = [], []

        for jf in sorted(f for f in os.listdir(sdir) if f.endswith(".json")):
            gen = jf.replace(".json","")
            ds  = HydraFakeTestDataset(os.path.join(sdir,jf), DATA_ROOT, IMAGE_ROOT, 224)
            if len(ds) == 0: continue

            ensemble_p = None
            for model, views, w in loaded:
                p, l = predict(model, ds, views, device)
                ensemble_p = p*w if ensemble_p is None else ensemble_p + p*w

            t   = youden_thresh(ensemble_p, l)
            a05 = accuracy_score(l, (ensemble_p>0.5).astype(int))
            ayd = accuracy_score(l, (ensemble_p>t).astype(int))
            ap  = average_precision_score(l, ensemble_p) if len(np.unique(l))>1 else -1

            results[split]["per_generator"][gen] = {
                "acc_05":float(a05),"acc_youden":float(ayd),"ap":float(ap),"n":len(ds)}
            all_p.extend(ensemble_p.tolist()); all_l.extend(l.tolist())
            print(f"  {gen:22s} ({split})  @0.5={a05*100:.1f}%  Youden={ayd*100:.1f}%  AP={ap:.4f}")

        p_arr, l_arr = np.array(all_p), np.array(all_l)
        t = youden_thresh(p_arr, l_arr)
        results[split]["acc_05"]    = float(accuracy_score(l_arr,(p_arr>0.5).astype(int)))
        results[split]["acc_youden"]= float(accuracy_score(l_arr,(p_arr>t).astype(int)))
        results[split]["ap"]        = float(average_precision_score(l_arr,p_arr))
        print(f"  {split.upper()} avg: @0.5={results[split]['acc_05']*100:.1f}%  "
              f"Youden={results[split]['acc_youden']*100:.1f}%\n")

    vals_05 = [results[s]["acc_05"]     for s in ["id","cm","cf","cd"]]
    vals_yd = [results[s]["acc_youden"] for s in ["id","cm","cf","cd"]]
    results["average"] = {"acc_05":float(np.mean(vals_05)), "acc_youden":float(np.mean(vals_yd))}

    print("="*70)
    print(f"  ID   {results['id']['acc_youden']*100:.1f}%  (v5: 86.9%  v14: 89.4%)")
    print(f"  CM   {results['cm']['acc_youden']*100:.1f}%  (v5: 99.5%  v14: 99.7%)")
    print(f"  CF   {results['cf']['acc_youden']*100:.1f}%  (v5: 91.7%  v14: 88.4%)")
    print(f"  CD   {results['cd']['acc_youden']*100:.1f}%  (v5: 80.6%  v14: 77.9%)")
    print(f"  AVG  {results['average']['acc_youden']*100:.1f}%  (v5: 89.7%  v14: 88.8%)")
    print("="*70)
    print(f"\n*** ENSEMBLE RESULT: {results['average']['acc_youden']*100:.2f}% (Youden) ***")

    json.dump(results, open(f"{OUTPUT_DIR}/test_results.json","w"), indent=2)
    print(f"Saved → {OUTPUT_DIR}/test_results.json")

if __name__ == "__main__":
    main()
