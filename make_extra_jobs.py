"""Job lists for strict zero-day (leave-one-family-out, world 1) and replication on two corpora (world 1)."""
fams = ["BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
jobs = []
for f in fams:
    for m in ["proposed", "mlp"]:
        jobs.append(f"--method {m} --seed 1 --holdout {f} --tag loafo --save_slots 0")
for ds in ["CIC-DIAD2024", "ToN-IoT"]:
    for m in ["proposed", "mlp", "gnn_gru", "gru", "lgbm"]:
        jobs.append(f"--dataset {ds} --method {m} --seed 1 --tag xdata --save_slots {int(m == 'proposed')}")
open("jobs_extra.txt", "w").write("\n".join(jobs))
print(len(jobs))
