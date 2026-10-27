#!/bin/bash
# Reproduces every experiment. Expects the three Parquet benchmark samples in ./data
# (CICIoT2023.parquet, CIC-DIAD2024.parquet, ToN-IoT.parquet). Results go to ./results.
set -e
cd "$(dirname "$0")"
mkdir -p results logs figures data
python3 stream_stats.py
python3 jobq.py jobs_main.txt 2          # backbones (seeds 1-5) and component studies (seeds 1-2)
python3 run_routing.py 1 2 3 4 5         # routing on the stored detector outputs
python3 bench_pq.py                       # post-quantum primitives, per-round overhead, tamper tests
python3 bench_eff.py                      # parameters, training step, streaming latency
python3 make_extra_jobs.py && python3 jobq.py jobs_extra.txt 2   # leave-one-family-out and replication
python3 compute_numbers.py                # every number reported in the paper -> numbers.json
python3 fig_arch.py && python3 fig_protocol.py
python3 -c "import fig_results as F; F.fig_backbones(['mlp','gru','attn','gnn','gnn_gru','gnn_attn','lgbm','proposed']); F.fig_reliability(); F.fig_firstseen(['fl_local','fl_fedrep','fl_fedavg','fl_split'], ['local only','FedRep heads','FedAvg','split sharing']); F.fig_efficiency('fig6_efficiency.png'); F.fig_loafo()"
