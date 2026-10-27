"""Cost of the post-quantum trust layer: primitive timings and sizes, per-round overhead of
signing and encrypting federated updates, and tamper-detection tests on the ledger."""
import json, os, time, statistics as stx
import numpy as np
from ledger import SIG, KEM, Ledger, merkle_root, merkle_proof, merkle_verify, session_key, H
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import x25519

RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
REPS = 200


def med(f, reps=REPS):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return stx.median(ts) * 1e3, np.percentile(ts, 95) * 1e3


def main():
    out = {"sig": {}, "kem": {}, "round": {}, "tamper": {}}
    msg = os.urandom(256)
    for name, alg in SIG.items():
        reps = 30 if "slh" in name else REPS
        pk, sk = alg.keygen()
        sig = alg.sign(sk, msg)
        kg = med(lambda: alg.keygen(), reps)
        sg = med(lambda: alg.sign(sk, msg), reps)
        vf = med(lambda: alg.verify(pk, msg, sig), reps)
        out["sig"][name] = dict(pk=len(pk), sig=len(sig), keygen_ms=kg[0], sign_ms=sg[0], sign_p95=sg[1],
                                verify_ms=vf[0], verify_p95=vf[1])
        print(name, out["sig"][name])
    for name, kem in KEM.items():
        pk, sk = kem.keygen()
        o = kem.encaps(pk)
        ct = o[0] if len(o[0]) == kem.CIPHERTEXT_SIZE else o[1]
        out["kem"][name] = dict(pk=len(pk), ct=len(ct), keygen_ms=med(lambda: kem.keygen())[0],
                                encaps_ms=med(lambda: kem.encaps(pk))[0], decaps_ms=med(lambda: kem.decaps(sk, ct))[0])
        print(name, out["kem"][name])
    a = x25519.X25519PrivateKey.generate(); b = x25519.X25519PrivateKey.generate()
    out["kem"]["x25519"] = dict(pk=32, ct=32, keygen_ms=med(lambda: x25519.X25519PrivateKey.generate())[0],
                                encaps_ms=med(lambda: a.exchange(b.public_key()))[0],
                                decaps_ms=med(lambda: a.exchange(b.public_key()))[0])
    # per-round overhead for one domain upload of the proposed model's shared parameters
    import jax, models as Mdl
    P = Mdl.init_params(jax.random.PRNGKey(0), 90, 9, 8, "gnn", "ssm")
    n_par = int(sum(np.size(x) for x in jax.tree_util.tree_leaves(P)))
    upd = np.random.default_rng(0).standard_normal(n_par).astype(np.float32).tobytes()
    out["n_params"] = n_par
    for sig_name in ["ed25519", "ml-dsa-44", "ml-dsa-65", "ed25519+ml-dsa-44"]:
        for kem_name in ["ml-kem-768"]:
            alg = SIG[sig_name]; pk, sk = alg.keygen()
            def one_round():
                key, kpk, kct = session_key(kem_name)
                nonce = os.urandom(12)
                ctxt = AESGCM(key).encrypt(nonce, upd, None)
                dig = H(ctxt)
                s = alg.sign(sk, dig)
                assert alg.verify(pk, dig, s)
                AESGCM(key).decrypt(nonce, ctxt, None)
                return len(ctxt) - len(upd) + kpk + kct + len(s) + 12
            extra = one_round()
            t = med(one_round, 60)
            out["round"][f"{sig_name}|{kem_name}"] = dict(ms=t[0], p95=t[1], extra_bytes=extra,
                                                          update_bytes=len(upd), rel_bytes=extra / len(upd))
            print(sig_name, kem_name, out["round"][f"{sig_name}|{kem_name}"])
    # ledger tamper detection: 6 domains x 9 checkpoints of alert batches + updates
    rng = np.random.default_rng(1)
    trials = 500
    def build(alg_name="ml-dsa-44"):
        L = Ledger(); keys = {}
        for dmn in range(6):
            pk, sk = SIG[alg_name].keygen(); L.enroll(dmn, alg_name, pk); keys[dmn] = sk
        for ck in range(9):
            for dmn in range(6):
                leaves = [os.urandom(48) for _ in range(40)]
                L.append(dmn, alg_name, keys[dmn], "alerts", leaves)
                L.append(dmn, alg_name, keys[dmn], "update", [H(os.urandom(64))])
        return L, keys
    L, keys = build()
    ok, _, _ = L.verify_chain()
    out["tamper"]["clean_ok"] = bool(ok)
    t = time.perf_counter(); L.verify_chain(); out["tamper"]["verify_chain_ms"] = (time.perf_counter() - t) * 1e3
    out["tamper"]["n_records"] = len(L.records)
    det = {"payload": 0, "header": 0, "forged_key": 0, "replay": 0}
    import copy, json as js
    for tr in range(trials):
        i = int(rng.integers(len(L.records)))
        # 1) payload byte flip
        L2 = copy.copy(L); L2.payloads = dict(L.payloads); pl = list(L.payloads[i]); j = int(rng.integers(len(pl)))
        b = bytearray(pl[j]); b[int(rng.integers(len(b)))] ^= 1; pl[j] = bytes(b); L2.payloads[i] = pl
        det["payload"] += not L2.verify_chain()[0]
        # 2) header edit (e.g., claimed domain or kind) without re-signing
        L3 = copy.copy(L); L3.records = list(L.records); r = dict(L.records[i]); h = js.loads(r["hdr"]); h["kind"] = "forged"
        r["hdr"] = js.dumps(h).encode(); L3.records[i] = r
        det["header"] += not L3.verify_chain()[0]
        # 3) record re-signed with an unenrolled key
        L4 = copy.copy(L); L4.records = list(L.records); r = dict(L.records[i]); pk2, sk2 = SIG["ml-dsa-44"].keygen()
        r["sig"] = SIG["ml-dsa-44"].sign(sk2, r["hdr"]); L4.records[i] = r
        det["forged_key"] += not L4.verify_chain()[0]
        # 4) replay of an earlier record in place of a later one
        if i > 0:
            L5 = copy.copy(L); L5.records = list(L.records); L5.payloads = dict(L.payloads)
            L5.records[i] = L.records[i - 1]; L5.payloads[i] = L.payloads[i - 1]
            det["replay"] += not L5.verify_chain()[0]
    out["tamper"]["trials"] = trials
    out["tamper"]["detected"] = det
    # merkle proof of one alert inside a signed batch
    leaves = L.payloads[0]; root = merkle_root(leaves); pf = merkle_proof(leaves, 7)
    out["tamper"]["merkle_proof_ok"] = bool(merkle_verify(leaves[7], pf, root))
    out["tamper"]["merkle_proof_bytes"] = len(pf) * 33
    json.dump(out, open(os.path.join(RES, "pq_bench.json"), "w"), indent=1)
    print(json.dumps(out["tamper"], indent=1))


if __name__ == "__main__":
    main()
