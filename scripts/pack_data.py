"""打包 data.tar.gz —— 包含 RAG 曲谱库所需的全部数据文件。"""
import tarfile
import os

DATA_DIRS = ["data/raw_midi", "data/rag", "data/curated_fingerstyle"]
OUTPUT = "data.tar.gz"

with tarfile.open(OUTPUT, "w:gz") as tar:
    for d in DATA_DIRS:
        if os.path.isdir(d):
            tar.add(d, arcname=d)
            print(f"  Added {d}")

size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
print(f"Done: {OUTPUT} ({size_mb:.1f} MB)")
