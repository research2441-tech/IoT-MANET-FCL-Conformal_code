"""Federated continual training and online evaluation protocol.

Timeline per domain (T slots, phase length T/4):
  * slots [0, T/4): unlabelled warm-up; federated self-supervised pretraining of the encoder
  * at checkpoint t_c (every `period` slots from T/4): audited labels of slots <= t_c - delay are
    known; the method decides whether to update, updates, recalibrates and then predicts
    slots [t_c, t_c + period) causally.
Audit: a random fraction of node-slots is labelled after `delay` slots; half of the audited
entries feed training, the other half calibration.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import models as Mdl
from conformal import class_pvalues, decide, VennAbers, JumperMartingale

L = Mdl.L_WIN
BURN = 16          # positions of a window that only provide context (loss on the last L-BURN)


class Method:
    def __init__(self, name, graph="gnn", temporal="ssm", fl="split", update="trigger", replay=True,
                 ssl=True, evidence=True, prox_mu=0.0, local_norm=True, lgbm=False):
        self.__dict__.update(locals())
        self.__dict__.pop("self")


METHODS = {
    # backbones (all with split personalisation, triggered updates, replay)
    "proposed": Method("proposed"),
    "mlp": Method("mlp", graph="mlp", temporal="none"),
    "gru": Method("gru", graph="mlp", temporal="gru"),
    "attn": Method("attn", graph="mlp", temporal="attn"),
    "gnn": Method("gnn", graph="gnn", temporal="none"),
    "gnn_gru": Method("gnn_gru", graph="gnn", temporal="gru"),
    "gnn_attn": Method("gnn_attn", graph="gnn", temporal="attn"),
    "lgbm": Method("lgbm", graph="mlp", temporal="none", fl="central", lgbm=True, update="periodic"),
    # federated strategies (proposed backbone, periodic updates so that only the sharing differs)
    "fl_local": Method("fl_local", fl="local", update="periodic"),
    "fl_fedavg": Method("fl_fedavg", fl="fedavg", update="periodic"),
    "fl_fedprox": Method("fl_fedprox", fl="fedavg", update="periodic", prox_mu=0.01),
    "fl_fedrep": Method("fl_fedrep", fl="fedrep", update="periodic"),
    "fl_split": Method("fl_split", fl="split", update="periodic"),
    "fl_central": Method("fl_central", fl="central", update="periodic"),
    # continual strategies
    "cl_static": Method("cl_static", update="never"),
    "cl_finetune": Method("cl_finetune", update="periodic", replay=False),
    # ablations
    "ab_nossl": Method("ab_nossl", ssl=False),
    "ab_noevid": Method("ab_noevid", evidence=False),
    "ab_globalnorm": Method("ab_globalnorm", local_norm=False),
}


class Protocol:
    def __init__(self, world, method, seed=0, audit=0.2, delay=30, period=50, alpha=0.05,
                 init_rounds=6, init_steps=20, upd_rounds=1, upd_steps=15, batch=4, lr=2e-3,
                 holdout=None, buffer_per_class=400, mart_thresh=np.log(100.0), verbose=True):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.doms = world["domains"]
        self.K = world["n_classes"]
        self.T = self.doms[0]["X"].shape[0]
        self.N = self.doms[0]["X"].shape[1]
        self.t0 = self.T // 4
        self.rng = np.random.default_rng(seed + 7)
        self.key = jax.random.PRNGKey(seed)
        self.timing = {"train": 0.0, "infer": 0.0}
        self.comm = {"uploads": 0, "downloads": 0, "rounds": 0}
        self._prep()

    # ------------------------------------------------------------------ data preparation
    def _prep(self):
        m = self.method
        F = self.doms[0]["X"].shape[-1]
        self.evid_cols = np.array([F - 4, F - 3, F - 2, F - 1])
        allX = np.concatenate([d["X"][:self.t0].reshape(-1, F) for d in self.doms])
        g_mu, g_sd = allX.mean(0), allX.std(0) + 1e-3
        for di, d in enumerate(self.doms):
            if m.local_norm:
                Xr = d["X"][:self.t0].reshape(-1, F)
                mu, sd = Xr.mean(0), Xr.std(0) + 1e-3
            else:
                mu, sd = g_mu, g_sd
            Xn = np.clip((d["X"] - mu) / sd, -10, 10).astype(np.float32)
            if not m.evidence:
                Xn[..., self.evid_cols] = 0.0
            d["Xn"] = Xn
            d["En"] = np.clip(d["E"], -5, 5).astype(np.float32)
            y = d["y"].copy()
            aud = self.rng.random(y.shape) < self.audit
            half = self.rng.random(y.shape) < 0.5
            if self.holdout is not None:
                aud &= y != self.holdout                 # family never labelled (strict zero-day)
            d["lab_train"] = aud & half
            d["lab_cal"] = aud & ~half
            d["buffer"] = {}
            d["mart"] = JumperMartingale()
            d["alpha"] = self.alpha
            d["store"] = []
            d["seen"] = np.zeros(self.K, bool)
            d["seen"][0] = True
            d["last_upd"] = 0

    # ------------------------------------------------------------------ windows
    def _windows(self, d, ends, with_labels=True):
        idx = ends[:, None] - L + 1 + np.arange(L)[None, :]
        X = d["Xn"][idx]; E = d["En"][idx]; M = d["M"][idx]
        if not with_labels:
            return X, E, M
        Y = d["y"][idx]
        W = d["lab_train"][idx].astype(np.float32)
        W[:, :BURN] = 0
        return X, E, M, Y, W

    def _update_buffer(self, d, avail):
        """Class-wise reservoir memory of labelled training entries (t, v)."""
        lo = d.get("buf_t", -1) + 1
        if avail < lo:
            return
        tt, vv = np.nonzero(d["lab_train"][lo:avail + 1])
        tt = tt + lo
        for t, v in zip(tt, vv):
            c = int(d["y"][t, v])
            buf = d["buffer"].setdefault(c, [])
            d.setdefault("seen_n", {}).setdefault(c, 0)
            d["seen_n"][c] += 1
            if len(buf) < self.buffer_per_class:
                buf.append((t, v))
            else:
                j = self.rng.integers(d["seen_n"][c])
                if j < self.buffer_per_class:
                    buf[j] = (t, v)
        d["buf_t"] = avail

    def _sample_ends(self, d, t_now, n, recent_from):
        """Window ends whose last L-BURN positions contain labelled training entries. Half of the
        batch comes from entries labelled since the last update, half from the class-balanced
        reservoir (experience replay); without replay all of it comes from recent entries."""
        avail = t_now - self.delay
        lo = max(0, recent_from)
        tt, vv = np.nonzero(d["lab_train"][lo:avail + 1])
        tt = tt + lo
        ends = []
        for i in range(n):
            if self.method.replay and d["buffer"] and (i % 2 == 1 or len(tt) == 0):
                c = self.rng.choice(sorted(d["buffer"].keys()))
                t, v = d["buffer"][c][self.rng.integers(len(d["buffer"][c]))]
            elif len(tt):
                t = tt[self.rng.integers(len(tt))]
            else:
                t = avail
            t = t + self.rng.integers(0, L - BURN)
            ends.append(int(np.clip(t, L - 1, avail)))
        return np.array(ends)

    # ------------------------------------------------------------------ training
    def _class_mask(self, d):
        if self.method.fl in ("local", "fedrep"):
            return d["seen"].copy()
        return np.any([x["seen"] for x in self.doms], 0)

    def _cls_weights(self, d):
        cnt = np.bincount(d["y"][d["lab_train"]], minlength=self.K).astype(float) + 1
        w = 1 / np.sqrt(cnt)
        return (w / w[0]).astype(np.float32)

    def _make_opt(self):
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.lr, weight_decay=1e-4))

    def _build_steps(self):
        opt = self._make_opt()
        mu = self.method.prox_mu

        def loss_fn(P, X, E, M, Y, W, cm, cw, Pg):
            l = Mdl.sup_loss(P, X, E, M, Y, W, cm, cw)
            if mu > 0:
                l = l + 0.5 * mu * sum(jnp.sum((a - b) ** 2) for a, b in
                                       zip(jax.tree_util.tree_leaves(P), jax.tree_util.tree_leaves(Pg)))
            return l

        @jax.jit
        def step(P, s, X, E, M, Y, W, cm, cw, Pg):
            l, g = jax.value_and_grad(loss_fn)(P, X, E, M, Y, W, cm, cw, Pg)
            u, s = opt.update(g, s, P)
            return optax.apply_updates(P, u), s, l

        def ssl_l(P, X, E, M, k):
            return Mdl.ssl_loss(P, X, E, M, k)

        @jax.jit
        def ssl_step(P, s, X, E, M, k):
            l, g = jax.value_and_grad(ssl_l)(P, X, E, M, k)
            u, s = opt.update(g, s, P)
            return optax.apply_updates(P, u), s, l

        @jax.jit
        def enc(P, X, E, M):
            return Mdl.graph_encode(P, X, E, M)

        @jax.jit
        def head(P, H, cm):
            h = Mdl.temporal(P, H)
            z = Mdl.ln(P["head_ln"], h[:, -1])
            lg = Mdl.lin(P["head"], z)
            return jnp.where(cm, lg, -1e9), z

        self.opt = opt
        self._step, self._ssl_step, self._enc, self._head = step, ssl_step, enc, head

    def _local_train(self, P, d, t_now, steps, recent_from, Pg):
        s = self.opt.init(P)
        self._update_buffer(d, t_now - self.delay)
        cm = jnp.asarray(self._class_mask(d)); cw = jnp.asarray(self._cls_weights(d))
        for _ in range(steps):
            ends = self._sample_ends(d, t_now, self.batch, recent_from)
            X, E, M, Y, W = self._windows(d, ends)
            P, s, l = self._step(P, s, X, E, M, Y, W, cm, cw, Pg)
        return P

    def _ssl_pretrain(self, P, rounds=6, steps=25):
        s_all = None
        for r in range(rounds):
            locs = []
            for d in self.doms:
                s = self.opt.init(P)
                Pl = P
                for _ in range(steps):
                    ts = self.rng.integers(0, self.t0, 16)
                    self.key, k = jax.random.split(self.key)
                    Pl, s, l = self._ssl_step(Pl, s, d["Xn"][ts], d["En"][ts], d["M"][ts], k)
                locs.append(Pl)
            P = jax.tree_util.tree_map(lambda *a: sum(a) / len(a), *locs)
        return P

    # ------------------------------------------------------------------ aggregation
    def _aggregate(self, locals_, weights, prev_locals):
        w = np.asarray(weights, float); w = w / w.sum()
        G = jax.tree_util.tree_map(lambda *a: sum(wi * ai for wi, ai in zip(w, a)), *locals_)
        out = []
        for Pl in prev_locals:
            Q = jax.tree_util.tree_map(lambda x: x, G)
            if self.method.fl == "split":
                Q["head"] = {"w": G["head"]["w"].at[:, 0].set(Pl["head"]["w"][:, 0]),
                             "b": G["head"]["b"].at[0].set(Pl["head"]["b"][0])}
            elif self.method.fl == "fedrep":
                Q["head"] = Pl["head"]; Q["head_ln"] = Pl["head_ln"]
            out.append(Q)
        return G, out

    # ------------------------------------------------------------------ inference
    def _infer(self, P, d, slots):
        """Causal windowed inference for the given slots -> softmax probs, embeddings, logits."""
        cm = jnp.asarray(self._class_mask(d))
        lo, hi = int(slots.min()) - L + 1, int(slots.max()) + 1
        H = []
        for a in range(lo, hi, 32):
            b = min(hi, a + 32)
            H.append(np.asarray(self._enc(P, d["Xn"][a:b], d["En"][a:b], d["M"][a:b])))
        H = np.concatenate(H)
        probs, embs = [], []
        for i in range(0, len(slots), 16):
            ss = slots[i:i + 16]
            idx = ss[:, None] - L + 1 + np.arange(L)[None, :] - lo
            lg, z = self._head(P, H[idx], cm)
            probs.append(np.asarray(jax.nn.softmax(lg, -1))); embs.append(np.asarray(z))
        return np.concatenate(probs), np.concatenate(embs)

    # ------------------------------------------------------------------ calibration
    def _maha_stats(self, embs_by_dom, labels_by_dom, local):
        """Class means (benign per domain, attacks pooled unless local) and a shared precision."""
        stats = []
        allz = np.concatenate(embs_by_dom); ally = np.concatenate(labels_by_dom)
        for z, y in zip(embs_by_dom, labels_by_dom):
            zz, yy = (z, y) if local else (allz, ally)
            mu = np.zeros((self.K, zz.shape[1]))
            for c in range(self.K):
                src = (z[y == c] if c == 0 else zz[yy == c])
                if len(src) == 0:
                    src = allz[ally == c] if (ally == c).any() else np.zeros((1, zz.shape[1]))
                mu[c] = src.mean(0)
            Zc = np.concatenate([zz - mu[yy]] + ([z - mu[y]] if not local else []))
            cov = np.cov(Zc.T) if len(Zc) > 2 else np.eye(zz.shape[1])
            cov = 0.9 * cov + 0.1 * np.trace(cov) / len(cov) * np.eye(len(cov))
            stats.append((mu, np.linalg.inv(cov)))
        return stats

    @staticmethod
    def _maha(z, st):
        mu, prec = st
        diff = z[:, None, :] - mu[None]
        return np.sqrt(np.maximum(np.einsum("nkd,de,nke->nk", diff, prec, diff), 0))

    def _calibrate(self, params_local, t_c):
        """Nonconformity scores of calibration entries. Recent window = last 150 labelled slots;
        classes with fewer than 30 pooled recent calibration entries are topped up with up to 40
        older entries per domain. Class means use training-half entries from the same slots."""
        avail = t_c - self.delay
        lo = max(L - 1, avail - 150)
        pooled_recent = np.zeros(self.K)
        for d in self.doms:
            sl = np.arange(lo, avail + 1)
            pooled_recent += np.bincount(d["y"][sl][d["lab_cal"][sl]], minlength=self.K)
        cal_items = []
        for di, d in enumerate(self.doms):
            recent = np.arange(lo, avail + 1)
            extra = []
            for c in range(self.K):
                for key, need in (("lab_cal", pooled_recent[c] < 30), ("lab_train", True)):
                    if not need:
                        continue
                    tt, vv = np.nonzero(d[key][L - 1:lo] & (d["y"][L - 1:lo] == c))
                    if len(tt):
                        pick = self.rng.choice(len(tt), min(40, len(tt)), replace=False)
                        extra.append(tt[pick] + L - 1)
            hist = np.unique(np.concatenate(extra)) if extra else np.zeros(0, int)
            slots = np.concatenate([hist, recent])
            pr, z = self._infer(params_local[di], d, slots)
            is_hist = np.arange(len(slots)) < len(hist)
            yy = d["y"][slots]
            cal_m = d["lab_cal"][slots] & (~is_hist[:, None] | (pooled_recent[yy] < 30))
            tr_m = d["lab_train"][slots]
            cal_items.append(dict(prob=pr[cal_m], z=z[cal_m], y=yy[cal_m], ztr=z[tr_m], ytr=yy[tr_m]))
        local = self.method.fl in ("local", "fedrep")
        stats = self._maha_stats([c["ztr"] for c in cal_items], [c["ytr"] for c in cal_items], local)
        cals = []
        for di, c in enumerate(cal_items):
            soft = -np.log(np.clip(c["prob"], 1e-12, 1))
            maha = self._maha(c["z"], stats[di])
            cals.append(dict(soft=soft, maha=maha, y=c["y"]))
        return stats, cals

    def _cal_sets(self, cals, di, mondrian=True, pooled_attacks=True):
        """Per-class calibration score lists for domain di."""
        out = {}
        for c in range(self.K):
            if c == 0 and mondrian:
                src = [cals[di]]
            elif pooled_attacks and self.method.fl not in ("local", "fedrep"):
                src = cals
            else:
                src = [cals[di]]
            s = np.concatenate([x["soft"][x["y"] == c, c] for x in src])
            m = np.concatenate([x["maha"][x["y"] == c, c] for x in src])
            out[c] = (s, m)
        return out

    # ------------------------------------------------------------------ main loop
    def run(self):
        m = self.method
        if m.lgbm:
            return self._run_lgbm()
        self._build_steps()
        F = self.doms[0]["X"].shape[-1]
        P = Mdl.init_params(self.key, F, 9, self.K, m.graph, m.temporal)
        tic = time.time()
        if m.ssl and m.graph == "gnn":
            P = self._ssl_pretrain(P)
        self.timing["ssl"] = time.time() - tic
        for d in self.doms:
            d["seen"] |= np.bincount(d["y"][:self.t0 - self.delay + 1][d["lab_train"][:self.t0 - self.delay + 1]],
                                     minlength=self.K) > 0
        params = [P for _ in self.doms]
        checkpoints = list(range(self.t0, self.T, self.period))
        log = []
        for ci, t_c in enumerate(checkpoints):
            avail = t_c - self.delay
            # labels that became available since the last checkpoint
            for d in self.doms:
                newly = d["lab_train"][:avail + 1] | d["lab_cal"][:avail + 1]
                d["seen_now"] = np.bincount(d["y"][:avail + 1][newly], minlength=self.K) > 0
            # ---- update decision
            if ci == 0:
                upd = [True] * len(self.doms)
                rounds, steps = self.init_rounds, self.init_steps
            else:
                rounds, steps = self.upd_rounds, self.upd_steps
                if m.update == "never":
                    upd = [False] * len(self.doms)
                elif m.update == "periodic":
                    upd = [True] * len(self.doms)
                else:
                    upd = [self._trigger(d, t_c) for d in self.doms]
            for d in self.doms:
                d["seen"] |= d["seen_now"]
            tic = time.time()
            if any(upd):
                if m.fl == "central":
                    params = self._central_train(params[0], t_c, rounds * steps, ci)
                else:
                    for r in range(rounds):
                        Pg = params[0] if m.fl != "local" else None
                        locs = []
                        for di, d in enumerate(self.doms):
                            if upd[di]:
                                rf = d["last_upd"] if ci > 0 else 0
                                locs.append(self._local_train(params[di], d, t_c, steps, rf - self.delay,
                                                              params[di]))
                            else:
                                locs.append(params[di])
                        if m.fl == "local":
                            params = locs
                        else:
                            ws = [max(1, d["lab_train"][:avail + 1].sum()) if u else 0
                                  for d, u in zip(self.doms, upd)]
                            sel = [i for i, u in enumerate(upd) if u]
                            G, newp = self._aggregate([locs[i] for i in sel], [ws[i] for i in sel], locs)
                            params = newp
                            self.comm["uploads"] += len(sel)
                            self.comm["downloads"] += len(self.doms)
                            self.comm["rounds"] += 1
                for di, d in enumerate(self.doms):
                    if upd[di]:
                        d["last_upd"] = t_c
                        d["mart"].reset()
            self.timing["train"] += time.time() - tic
            # ---- calibrate and predict the next block
            tic = time.time()
            stats, cals = self._calibrate(params, t_c)
            for di, d in enumerate(self.doms):
                slots = np.arange(t_c, min(self.T, t_c + self.period))
                pr, z = self._infer(params[di], d, slots)
                soft = -np.log(np.clip(pr, 1e-12, 1))
                maha = self._maha(z.reshape(-1, z.shape[-1]), stats[di]).reshape(z.shape[0], z.shape[1], -1)
                d["store"].append(dict(t_c=t_c, slots=slots, prob=pr.astype(np.float32),
                                       soft=soft.astype(np.float32), maha=maha.astype(np.float32),
                                       active=self._class_mask(d), cal=cals, di=di, updated=upd[di]))
            self.timing["infer"] += time.time() - tic
            log.append(dict(t_c=t_c, upd=[bool(u) for u in upd]))
            if self.verbose:
                print(f"  t={t_c} upd={sum(upd)} train={self.timing['train']:.0f}s infer={self.timing['infer']:.0f}s",
                      flush=True)
        self.params = params
        self._seen_global = np.any([x["seen"] for x in self.doms], 0)
        return log

    def deploy(self, world):
        """Apply the trained (frozen) models to a new world: local normalisation and conformal
        recalibration from the new world's audited labels, no parameter updates."""
        trained = self.params
        self.doms = world["domains"]
        self.T = self.doms[0]["X"].shape[0]; self.N = self.doms[0]["X"].shape[1]
        self._prep()
        n_old = len(trained)
        params = [trained[i % n_old] for i in range(len(self.doms))]
        for ci, t_c in enumerate(range(self.t0, self.T, self.period)):
            avail = t_c - self.delay
            for d in self.doms:
                lab = d["lab_train"][:avail + 1] | d["lab_cal"][:avail + 1]
                d["seen"] = self._seen_global.copy() if hasattr(self, "_seen_global") else d["seen"]
            stats, cals = self._calibrate(params, t_c)
            for di, d in enumerate(self.doms):
                slots = np.arange(t_c, min(self.T, t_c + self.period))
                pr, z = self._infer(params[di], d, slots)
                maha = self._maha(z.reshape(-1, z.shape[-1]), stats[di]).reshape(z.shape[0], z.shape[1], -1)
                d["store"].append(dict(t_c=t_c, slots=slots, prob=pr.astype(np.float32),
                                       soft=(-np.log(np.clip(pr, 1e-12, 1))).astype(np.float32),
                                       maha=maha.astype(np.float32), active=self._class_mask(d), cal=cals,
                                       di=di, updated=False))

    def _trigger(self, d, t_c):
        """Drift trigger. Fires when (i) audited labels reveal a family the deployed model does not
        know, or (ii) a Simple Jumper conformal test martingale over the benign-class p-values of
        newly audited benign entries exceeds the threshold (local normality has drifted)."""
        avail = t_c - self.delay
        lo = max(self.t0, avail - self.period + 1)
        new_class = False
        fired = False
        pv = {}      # node -> benign p-values of its audited benign entries in the new label window
        for st in d["store"]:
            sl = st["slots"]
            sel = (sl >= lo) & (sl <= avail)
            if not sel.any():
                continue
            cal = self._cal_sets(st["cal"], st["di"])
            for i in np.where(sel)[0]:
                t = sl[i]
                lab = d["lab_train"][t] | d["lab_cal"][t]
                yv = d["y"][t]
                if (lab & ~st["active"][yv]).any():
                    new_class = True
                vs = np.where(lab & (yv == 0))[0]
                if len(vs) == 0:
                    continue
                Pv = class_pvalues(st["soft"][i, vs], st["maha"][i, vs], {0: cal[0]},
                                   np.eye(self.K, dtype=bool)[0])
                for v, p in zip(vs, Pv[:, 0]):
                    pv.setdefault(int(v), []).append(float(p))
        # thinning: one randomly chosen p-value per node, because successive windows of a node are
        # strongly dependent and the martingale is only valid for exchangeable sequences
        for v in self.rng.permutation(sorted(pv)):
            p = pv[v][self.rng.integers(len(pv[v]))]
            if d["mart"].update(p) > self.mart_thresh:
                fired = True
        d.setdefault("trig_log", []).append((int(t_c), bool(new_class), bool(fired), float(d["mart"].logM)))
        return new_class or fired

    def _central_train(self, P, t_c, steps, ci):
        s = self.opt.init(P)
        cm = jnp.asarray(np.any([x["seen"] for x in self.doms], 0))
        cnt = sum(np.bincount(d["y"][d["lab_train"]], minlength=self.K) for d in self.doms) + 1.0
        cw = jnp.asarray((1 / np.sqrt(cnt) / (1 / np.sqrt(cnt[0]))).astype(np.float32))
        for d in self.doms:
            self._update_buffer(d, t_c - self.delay)
        for _ in range(steps * len(self.doms) // 2):
            d = self.doms[self.rng.integers(len(self.doms))]
            rf = d["last_upd"] if ci > 0 else 0
            ends = self._sample_ends(d, t_c, self.batch, rf - self.delay)
            X, E, M, Y, W = self._windows(d, ends)
            P, s, l = self._step(P, s, X, E, M, Y, W, cm, cw, P)
        return [P for _ in self.doms]

    # ------------------------------------------------------------------ LightGBM reference
    def _run_lgbm(self):
        import lightgbm as lgb
        checkpoints = list(range(self.t0, self.T, self.period))
        for ci, t_c in enumerate(checkpoints):
            avail = t_c - self.delay
            Xs, ys = [], []
            for d in self.doms:
                lab = d["lab_train"][:avail + 1]
                Xs.append(d["Xn"][:avail + 1][lab]); ys.append(d["y"][:avail + 1][lab])
                d["seen"] |= np.bincount(d["y"][:avail + 1][lab | d["lab_cal"][:avail + 1]], minlength=self.K) > 0
            Xa, ya = np.concatenate(Xs), np.concatenate(ys)
            classes = np.unique(ya)
            tic = time.time()
            clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, class_weight="balanced",
                                     verbose=-1, n_jobs=2)
            clf.fit(Xa, ya)
            self.timing["train"] += time.time() - tic

            def prob_full(Xq):
                p = np.zeros((len(Xq), self.K)); p[:, clf.classes_] = clf.predict_proba(Xq); return p
            lo = max(L - 1, avail - 150)
            cals = []
            for d in self.doms:
                sl = np.arange(lo, avail + 1)
                msk = d["lab_cal"][sl]
                pr = prob_full(d["Xn"][sl][msk])
                cals.append(dict(soft=-np.log(np.clip(pr, 1e-12, 1)), maha=np.zeros_like(pr), y=d["y"][sl][msk]))
            for di, d in enumerate(self.doms):
                slots = np.arange(t_c, min(self.T, t_c + self.period))
                Xq = d["Xn"][slots].reshape(-1, d["Xn"].shape[-1])
                pr = prob_full(Xq).reshape(len(slots), self.N, self.K)
                act = np.zeros(self.K, bool); act[classes] = True
                d["store"].append(dict(t_c=t_c, slots=slots, prob=pr.astype(np.float32),
                                       soft=-np.log(np.clip(pr, 1e-12, 1)).astype(np.float32), maha=None,
                                       active=act, cal=cals, di=di, updated=True))
        return []
