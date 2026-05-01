#!/bin/bash
# XGenDet Dataset Download Script
# Downloads additional datasets for training and evaluation

set -e

DATA_ROOT=${1:-"/home/sachin.chaudhary/datasets"}
mkdir -p "$DATA_ROOT"

echo "============================================"
echo "XGenDet Dataset Download"
echo "Target: $DATA_ROOT"
echo "============================================"

# 1. GenImage Dataset
echo ""
echo "--- GenImage ---"
echo "GenImage must be downloaded manually from:"
echo "  https://github.com/GenImage-Dataset/GenImage"
echo "Place the downloaded files in: $DATA_ROOT/GenImage/"
echo ""

# 2. DF40 Benchmark
echo "--- DF40 ---"
if [ ! -d "$DATA_ROOT/DF40" ]; then
    echo "Cloning DF40 repository..."
    git clone https://github.com/YZY-stack/DF40 "$DATA_ROOT/DF40"
    echo "Follow instructions in $DATA_ROOT/DF40/README.md to download data"
else
    echo "DF40 already exists, skipping."
fi

# 3. Community Forensics
echo ""
echo "--- Community Forensics (GAPL dataset) ---"
echo "Download via HuggingFace:"
echo "  pip install datasets"
echo "  python -c \"from datasets import load_dataset; ds = load_dataset('CommunityForensics/CommunityForensics', cache_dir='$DATA_ROOT/community_forensics')\""
echo ""

# 4. Verify existing data
echo "--- Verifying existing GTA data ---"
GTA_ROOT="/home/sachin.chaudhary/GTA"

if [ -d "$GTA_ROOT/final_GENERATORS" ]; then
    echo "In-domain generators found:"
    ls "$GTA_ROOT/final_GENERATORS/" 2>/dev/null | head -20
    echo ""
fi

if [ -d "$GTA_ROOT/OOD_GENERATORS" ]; then
    echo "Out-of-domain generators found:"
    ls "$GTA_ROOT/OOD_GENERATORS/" 2>/dev/null | head -20
    echo ""
fi

echo "============================================"
echo "Dataset setup checklist:"
echo "  [x] GTA ID generators (local)"
echo "  [x] GTA OOD generators (local)"
echo "  [ ] GenImage (manual download)"
echo "  [ ] DF40 (cloned, needs data download)"
echo "  [ ] Community Forensics (HuggingFace)"
echo "============================================"
