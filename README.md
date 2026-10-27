# Federated continual IoT-MANET intrusion detection with node memory and conformal calibration

Code and per-run results for the manuscript "Recurrent Node Memory, Conformal Risk Calibration and
Post-Quantum Provenance for Federated Continual Zero-Day Intrusion Detection in IoT-MANETs".

## Data
Place the capped Parquet samples of CICIoT2023, CIC IoT-DIAD 2024 and TON_IoT (from
https://github.com/JenefaNaveen09/TokenGauss-Former) in `./data`, or set `CFG_DATA` to their folder.

## Files
| File | Purpose |
|---|---|
| data.py | record pools, chronological chunks, device profiles |
| sim.py | IoT-MANET stream simulator (mobility, on-off attackers, watchdog evidence, temporal graphs) |
| models.py | edge-gated graph encoder, selective state-space, GRU and attention memories, losses (JAX) |
| fcl.py | federated continual protocol: SSL pretraining, split/FedAvg/FedProx/FedRep/local/central sharing, reservoir replay, drift trigger, calibration |
| conformal.py | conformal p-values, dual decision rule, Venn-Abers predictor, Simple Jumper martingale, ECE |
| evaluate.py | offline scoring of stored predictions (ACI, variants, metrics) |
| routing.py, run_routing.py | packet-level routing with the policies of the paper |
| ledger.py, bench_pq.py | hash-chained ledger, ML-DSA/SLH-DSA/Ed25519 signatures, ML-KEM, tamper tests |
| bench_eff.py | parameter counts, training step time, streaming latency |
| run.py, jobq.py, jobs_main.txt, jobs_extra.txt, make_extra_jobs.py | experiment runner and job lists |
| stream_stats.py, analyze.py, compute_numbers.py | statistics and every number in the paper |
| diagram.py, fig_arch.py, fig_protocol.py, fig_results.py | figures |
| all_run_results.json | metrics of every run (keyed by file name) |
| numbers.json | aggregated numbers used in the manuscript |
| references_verified.json | reference list with verification source |

## Reproduce
`bash run_all.sh` (about 20 hours on two CPU cores). Every run is deterministic given its seed.
Development world: seed 101. Evaluation worlds: seeds 1-5.
