#!/usr/bin/env bash
set -euo pipefail
python3 /mnt/sevenT/wrz_data/gofa-demo/run_gofa.py --override /mnt/sevenT/wrz_data/GOFA/cache_data/gofa_cache_exp/large_h6_n10_s100_wikics_arxiv/configs/wikics/formal_trace.yaml
python3 /mnt/sevenT/wrz_data/gofa-demo/run_gofa.py --override /mnt/sevenT/wrz_data/GOFA/cache_data/gofa_cache_exp/large_h6_n10_s100_wikics_arxiv/configs/arxiv/formal_trace.yaml
