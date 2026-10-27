"""Verifiable trust ledger with crypto-agile post-quantum signatures and key establishment.

Records (alert batches, federated model updates, calibration digests) are hash-chained and
signed by the enrolled gateway of each MANET domain. The algorithm identifier travels in the
signed header, so verifiers dispatch on it and an algorithm can be replaced without changing
the record format. Payloads are committed by a Merkle root, which lets a single alert be proven
against a signed batch.
"""
import hashlib, json, os, struct, time
import numpy as np
from pqcrypto.sign import ml_dsa_44, ml_dsa_65, ml_dsa_87, slh_dsa_sha2_128f
from pqcrypto.kem import ml_kem_512, ml_kem_768, ml_kem_1024
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization

H = lambda b: hashlib.sha3_256(b).digest()


class Ed25519Alg:
    name = "ed25519"
    def keygen(self):
        sk = ed25519.Ed25519PrivateKey.generate()
        return sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw), sk
    def sign(self, sk, m): return sk.sign(m)
    def verify(self, pk, m, s):
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(pk).verify(s, m); return True
        except Exception:
            return False


class PQAlg:
    def __init__(self, mod, name): self.m, self.name = mod, name
    def keygen(self): return self.m.keygen()
    def sign(self, sk, m): return self.m.sign(sk, m)
    def verify(self, pk, m, s):
        try:
            self.m.verify(pk, m, s); return True
        except Exception:
            return False


class HybridAlg:
    """Ed25519 + ML-DSA-44: both signatures must verify."""
    name = "ed25519+ml-dsa-44"
    def __init__(self): self.a, self.b = Ed25519Alg(), PQAlg(ml_dsa_44, "ml-dsa-44")
    def keygen(self):
        pa, sa = self.a.keygen(); pb, sb = self.b.keygen(); return pa + pb, (sa, sb)
    def sign(self, sk, m): return self.a.sign(sk[0], m) + self.b.sign(sk[1], m)
    def verify(self, pk, m, s): return self.a.verify(pk[:32], m, s[:64]) and self.b.verify(pk[32:], m, s[64:])


SIG = {a.name: a for a in [Ed25519Alg(), PQAlg(ml_dsa_44, "ml-dsa-44"), PQAlg(ml_dsa_65, "ml-dsa-65"),
                           PQAlg(ml_dsa_87, "ml-dsa-87"), PQAlg(slh_dsa_sha2_128f, "slh-dsa-sha2-128f"), HybridAlg()]}
KEM = {"ml-kem-512": ml_kem_512, "ml-kem-768": ml_kem_768, "ml-kem-1024": ml_kem_1024}


def merkle_root(leaves):
    level = [H(b"\x00" + x) for x in leaves] or [H(b"")]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [H(b"\x01" + level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves, idx):
    level = [H(b"\x00" + x) for x in leaves]
    proof = []
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = idx ^ 1
        proof.append((level[sib], sib < idx))
        level = [H(b"\x01" + level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return proof


def merkle_verify(leaf, proof, root):
    h = H(b"\x00" + leaf)
    for sib, left in proof:
        h = H(b"\x01" + sib + h) if left else H(b"\x01" + h + sib)
    return h == root


class Ledger:
    def __init__(self):
        self.records = []
        self.enrolled = {}            # domain -> (alg, pk)
        self.payloads = {}

    def enroll(self, domain, alg, pk):
        self.enrolled[domain] = (alg, pk)

    def header(self, domain, alg, kind, root):
        prev = self.records[-1]["hash"] if self.records else b"\x00" * 32
        seq = len(self.records)
        return json.dumps(dict(v=1, alg=alg, dom=domain, kind=kind, seq=seq, prev=prev.hex(),
                               root=root.hex())).encode()

    def append(self, domain, alg, sk, kind, leaves):
        root = merkle_root(leaves)
        hdr = self.header(domain, alg, kind, root)
        sig = SIG[alg].sign(sk, hdr)
        rec = dict(hdr=hdr, sig=sig, hash=H(hdr + sig))
        self.records.append(rec)
        self.payloads[len(self.records) - 1] = list(leaves)
        return rec

    def verify_chain(self):
        prev = b"\x00" * 32
        for i, r in enumerate(self.records):
            h = json.loads(r["hdr"])
            if h["seq"] != i or bytes.fromhex(h["prev"]) != prev:
                return False, i, "chain"
            if h["dom"] not in self.enrolled or self.enrolled[h["dom"]][0] != h["alg"]:
                return False, i, "enrol"
            alg, pk = self.enrolled[h["dom"]]
            if not SIG[alg].verify(pk, r["hdr"], r["sig"]):
                return False, i, "sig"
            if merkle_root(self.payloads[i]) != bytes.fromhex(h["root"]):
                return False, i, "payload"
            if H(r["hdr"] + r["sig"]) != r["hash"]:
                return False, i, "hash"
            prev = r["hash"]
        return True, -1, "ok"


def session_key(kem_name):
    """Gateway <-> aggregator key establishment with ML-KEM, then HKDF to an AES-256-GCM key."""
    kem = KEM[kem_name]
    pk, sk = kem.keygen()
    out = kem.encaps(pk)
    ct, ss_enc = (out[0], out[1]) if len(out[0]) == kem.CIPHERTEXT_SIZE else (out[1], out[0])
    ss_dec = kem.decaps(sk, ct)
    assert ss_enc == ss_dec
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"cfg-fl-round").derive(ss_enc)
    return key, len(pk), len(ct)
