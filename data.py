"""Record pools from public IoT intrusion benchmarks.

Each benchmark is a capped, de-duplicated sample of the public release (CICIoT2023,
CIC IoT-DIAD 2024, ToN-IoT) with a source-order field. Records of every family are split
by source order into four chronological chunks that feed the four stream phases, so later
phases replay later traffic of the same family.
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_DIR = os.environ.get("CFG_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

BENIGN = {"CICIoT2023": "Benign", "CIC-DIAD2024": "Benign", "ToN-IoT": "normal"}
# columns removed because they encode testbed configuration rather than behaviour
DROP = {"CICIoT2023": ["Time_To_Live"], "CIC-DIAD2024": [], "ToN-IoT": []}
N_PHASE = 4


class Pools:
    def __init__(self, name="CICIoT2023", n_profiles=8, seed=0):
        df = pd.read_parquet(os.path.join(DATA_DIR, name + ".parquet"))
        meta = ["family", "fine", "order"]
        feats = [c for c in df.columns if c not in meta and c not in DROP[name]]
        X = df[feats].to_numpy(np.float64).copy()
        fam = df["family"].astype(str).to_numpy()
        order = df["order"].to_numpy()
        self.name = name
        self.feature_names = feats
        benign = BENIGN[name]
        self.families = [benign] + sorted(f for f in np.unique(fam) if f != benign)
        self.fam_index = {f: i for i, f in enumerate(self.families)}
        y = np.array([self.fam_index[f] for f in fam])
        # chronological chunk of each record within its family
        chunk = np.zeros(len(y), int)
        for c in range(len(self.families)):
            idx = np.where(y == c)[0]
            idx = idx[np.argsort(order[idx], kind="stable")]
            chunk[idx] = np.minimum(np.arange(len(idx)) * N_PHASE // len(idx), N_PHASE - 1)
        # heavy-tailed non-negative columns are log-compressed, then z-scored on phase-0 data
        nonneg = (X.min(0) >= 0) & (X.max(0) > 50)
        X[:, nonneg] = np.log1p(X[:, nonneg])
        ref = chunk == 0
        mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-6
        X = np.clip((X - mu) / sd, -8, 8).astype(np.float32)
        self.X, self.y, self.chunk = X, y, chunk
        self.dim = X.shape[1]
        # compact record summary for edge attributes (fitted on phase-0 records)
        self.pca = PCA(n_components=4, random_state=seed).fit(X[ref])
        self.Z = self.pca.transform(X).astype(np.float32)
        # benign device profiles: k-means on phase-0 benign records, later records by nearest centroid
        b0 = np.where((y == 0) & ref)[0]
        km = KMeans(n_clusters=n_profiles, n_init=4, random_state=seed).fit(X[b0])
        bi = np.where(y == 0)[0]
        prof = km.predict(X[bi])
        self.n_profiles = n_profiles
        self.benign_pool = {(p, k): bi[(prof == p) & (chunk[bi] == k)] for p in range(n_profiles)
                            for k in range(N_PHASE)}
        # guarantee non-empty profile pools (fall back to all benign of that chunk)
        for (p, k), v in list(self.benign_pool.items()):
            if len(v) < 20:
                self.benign_pool[(p, k)] = bi[chunk[bi] == k]
        self.attack_pool = {(c, k): np.where((y == c) & (chunk == k))[0]
                            for c in range(1, len(self.families)) for k in range(N_PHASE)}

    def sample_benign(self, rng, profile, phase, n):
        pool = self.benign_pool[(profile, phase)]
        return pool[rng.integers(0, len(pool), n)]

    def sample_attack(self, rng, fam, phase, n):
        pool = self.attack_pool[(fam, phase)]
        if len(pool) == 0:
            pool = np.concatenate([self.attack_pool[(fam, k)] for k in range(N_PHASE)])
        return pool[rng.integers(0, len(pool), n)]


_CACHE = {}


def get_pools(name="CICIoT2023"):
    if name not in _CACHE:
        _CACHE[name] = Pools(name)
    return _CACHE[name]
