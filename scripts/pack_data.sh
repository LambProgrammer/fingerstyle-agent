#!/bin/bash
# 打包 data.tar.gz —— 包含 RAG 曲谱库所需的全部数据文件。
# 用法：bash scripts/pack_data.sh
# 产出：data.tar.gz（上传至 GitHub Release）

set -e

echo "打包 data.tar.gz ..."
tar -czf data.tar.gz \
  data/raw_midi/ \
  data/rag/ \
  data/curated_fingerstyle/

echo "完成：$(ls -lh data.tar.gz | awk '{print $5}')"
