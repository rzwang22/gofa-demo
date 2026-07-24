#!/usr/bin/env bash
set -euo pipefail
python3 /mnt/sevenT/wrz_data/gofa-demo/run_gofa.py --override /mnt/sevenT/wrz_data/GOFA/cache_data/gofa_cache_exp/split_suites/large_h6_n32_s100/configs/cora_node/full_cache.yaml
python3 /mnt/sevenT/wrz_data/gofa-demo/run_gofa.py --override /mnt/sevenT/wrz_data/GOFA/cache_data/gofa_cache_exp/split_suites/large_h6_n32_s100/configs/cora_link/full_cache.yaml
python3 /mnt/sevenT/wrz_data/gofa-demo/run_gofa.py --override /mnt/sevenT/wrz_data/GOFA/cache_data/gofa_cache_exp/split_suites/large_h6_n32_s100/configs/pubmed_node/full_cache.yaml
