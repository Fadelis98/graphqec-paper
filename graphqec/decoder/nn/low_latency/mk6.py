"""Host side of the low-latency cooperative GPU kernel (deterministic tile route): weight export, op-program builder, launchers.

Identical to the source used for the paper's RTX 5080 timing runs except that ``split`` is defined here.
"""
import ctypes
import math
import struct

import numpy as np
import torch


def split(dec, raw):
    """raw: (B, n_det) long tensor -> (enc, cycles, readout) views exactly as QECCDecoder._decode."""
    ni, nc = int(dec.num_init_check), int(dec.num_cycle_check)
    return raw[:, :ni], raw[:, ni:-ni].reshape(raw.shape[0], -1, nc), raw[:, -ni:]

OP_GEMM, OP_ATTN, OP_EMBED, OP_SCAT_MID, OP_GDN, OP_SCAT_LOG, OP_FINAL = 1, 2, 3, 4, 5, 6, 7
EPI_STORE, EPI_RESID, EPI_SWIGLU, EPI_RESID_TANH, EPI_MIX = 0, 1, 2, 3, 4
PRO_NONE, PRO_RMSNORM = 0, 1
CFG = {0: (32, 32), 1: (64, 64), 2: (128, 64), 3: (16, 64)}
BF = torch.bfloat16


def fbits(x):
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


def P(t):
    return 0 if t is None else int(t.data_ptr())


def unwrap(m):
    return getattr(m, "_orig_mod", m)


def rec(**kv):
    r = np.zeros(32, np.int64)
    for k, v in kv.items():
        r[int(k[1:])] = v
    return r


LADDER = [3, 0, 6, 1, 4, 5, 2, 7]
RES_LADDER = [3, 0, 6, 1, 4]
CFG.update({4: (32, 128), 5: (64, 128), 6: (16, 128), 7: (128, 128)})
OP_ZERO = 8
OP_NORM = 9
PRO_RESADD = 2
SMEM = 96256
MAXPH, MAXOPS = 256, 448  # must match mk6.cu


def res_smem(cfg, K, resadd, naux=3):
    bm, bn = CFG[cfg]
    return (bm + bn) * (K + 8) * 2 + (bm * K * 4 if resadd else 0) + 3 * K * 2 + bn * 2 + naux * bm * bn * 2 + bm * 4


class Prog:
    """Phases of op records; tile sizes / query chunks are chosen per phase so the phase fits ~one SM wave."""

    def __init__(self, n_sm=84, fixed_cfg=None, wave=1.0, lat_m=1536, allow_res=True, fuse_attn=True):
        self.phases = []
        self.labels = []
        self.keep = []
        self.n_sm = n_sm
        self.fixed_cfg = fixed_cfg
        self.wave = wave
        self.lat_m = lat_m
        self.allow_res = allow_res
        self.fuse_attn = fuse_attn

    @staticmethod
    def _items(s, choice):
        if s["kind"] == "gemm":
            bm, bn = CFG[choice]
            return math.ceil(s["M"] / bm) * math.ceil(s["N"] / bn)
        if s["kind"] == "attn":
            mt = math.ceil(s["gsz"] / 16)
            return s["ng"] * math.ceil(s["H"] / s["hb"]) * math.ceil(mt / choice)
        return 0

    def _ladder(self, s):
        if s["res"]:
            return [c for c in RES_LADDER if res_smem(c, s["K"], s["resadd"]) <= SMEM]
        return LADDER

    def _resolve(self, specs):
        choice = {}
        for i, s in enumerate(specs):
            if isinstance(s, np.ndarray):
                continue
            if s["kind"] == "gemm":
                if s["cfg"] is not None:
                    choice[i] = s["cfg"]
                elif self.fixed_cfg is not None and not s["res"]:
                    choice[i] = self.fixed_cfg
                else:
                    lad = self._ladder(s)
                    choice[i] = lad[0] if (s["res"] or s["M"] <= 16) else 0
            elif s["kind"] == "attn":
                choice[i] = math.ceil(s["gsz"] / 16) if s["whole"] else 1
        budget = int(self.n_sm * self.wave)
        while sum(self._items(specs[i], c) for i, c in choice.items()) > budget:
            best = None
            for i, c in choice.items():
                s = specs[i]
                cur = self._items(s, c)
                if s["kind"] == "gemm":
                    if s["cfg"] is not None or (self.fixed_cfg is not None and not s["res"]):
                        continue
                    lad = self._ladder(s)
                    nxt = next((x for x in lad[lad.index(c) + 1:] if self._items(s, x) < cur), None)
                else:
                    mt = math.ceil(s["gsz"] / 16)
                    nxt = next((x for x in range(c + 1, mt + 1) if self._items(s, x) < cur), None) if not s["whole"] else None
                if nxt is not None and (best is None or cur > best[2]):
                    best = (i, nxt, cur)
            if best is None:
                break
            choice[best[0]] = best[1]
        out = []
        for i, s in enumerate(specs):
            if isinstance(s, np.ndarray):
                out.append(s)
                continue
            r = s["rec"]
            if s["kind"] == "gemm":
                bm, bn = CFG[choice[i]]
                r[3], r[22] = choice[i], math.ceil(s["N"] / bn)
                r[2] = math.ceil(s["M"] / bm) * r[22]
                r[23] = 1 if s["res"] else 0
            else:
                r[20], r[21] = s["hb"], choice[i]
                r[2] = self._items(s, choice[i])
            out.append(r)
        return out

    def phase(self, ops, label=""):
        ops = [o for o in ops if o is not None]
        ops = [o for o in self._resolve(ops) if o[2] > 0]
        if ops:
            self.phases.append(ops)
            self.labels.append(label)

    def buf(self, rows, cols, dtype=BF):
        t = torch.zeros(rows, cols, dtype=dtype, device="cuda")
        self.keep.append(t)
        return t

    def finalize(self):
        recs, pb, ends, totals = [], [0], [], []
        for ops in self.phases:
            tb = 0
            for o in ops:
                o[1] = tb
                tb += o[2]
                recs.append(o)
                ends.append(tb)
            pb.append(len(recs))
            totals.append(tb)
        self.ops_t = torch.from_numpy(np.stack(recs)).cuda()
        self.pb_t = torch.tensor(pb, dtype=torch.int32, device="cuda")
        self.end_t = torch.tensor(ends, dtype=torch.int32, device="cuda")
        self.tot_t = torch.tensor(totals, dtype=torch.int32, device="cuda")
        self.n_phases = len(self.phases)
        self.n_items = totals
        assert self.n_phases <= MAXPH and len(recs) <= MAXOPS and max(totals) < 2 ** 31
        return self

    def mem_bytes(self):
        seen = {}
        for t in self.keep:
            s = t.untyped_storage()
            seen[s.data_ptr()] = s.nbytes()
        return int(sum(seen.values()))

    # ---------------------------------------------------------------- op constructors
    def can_res(self, M, K, resadd=False):
        return self.allow_res and M <= self.lat_m and any(res_smem(c, K, resadd) <= SMEM for c in RES_LADDER)

    def gemm(self, M, N, K, A, lda, W, bias, out, ldo, epi, norm=None, aux1=None, ldr=0, aux2=None, aux3=None, cfg=None,
             res=None, resadd=None):
        assert K % 64 == 0 and lda % 8 == 0 and W.is_contiguous() and W.shape == (N, K), (M, N, K, W.shape)
        assert norm is None or K <= 256
        res = self.can_res(M, K, resadd is not None) if res is None else res
        assert resadd is None or res
        pro = PRO_NONE if norm is None else (PRO_RESADD if resadd is not None else PRO_RMSNORM)
        nw, nb, eps = norm if norm is not None else (None, None, 0.0)
        r = rec(f0=OP_GEMM, f4=M, f5=N, f6=K, f7=P(A), f8=lda, f9=P(W), f10=P(bias), f11=P(out), f12=ldo, f13=P(aux1),
                f14=ldr, f15=P(nw), f16=P(nb), f17=fbits(eps), f18=epi, f19=pro, f20=P(aux2), f21=P(aux3))
        if resadd is not None:
            acc, ldacc, bo, x1, ldx1 = resadd
            r[24], r[25], r[26], r[27], r[28] = P(acc), ldacc, P(bo), P(x1), ldx1
        return {"kind": "gemm", "rec": r, "M": M, "N": N, "K": K, "cfg": cfg, "res": bool(res), "resadd": resadd is not None}

    def attn(self, qkv, E, hd, H, out, row0, gsz, ngroups, wo=None, whole=False):
        assert gsz <= 144 and hd in (24, 32) and not (whole and wo is not None)
        r = rec(f0=OP_ATTN, f4=P(qkv), f5=3 * E, f6=E, f7=hd, f8=H, f9=P(out), f10=E, f11=row0, f12=gsz, f13=ngroups,
                f15=fbits(hd ** -0.5), f17=1 if wo is not None else 0, f18=P(wo))
        hb = 2 if (whole and gsz <= 144) else 1
        return {"kind": "attn", "rec": r, "gsz": gsz, "ng": ngroups, "H": H, "whole": whole, "hb": hb}

    def norm(self, X, ldx, Xn, ldn, M, K, norm):
        assert K <= 256 and K % 8 == 0
        g, b, eps = norm
        return rec(f0=OP_NORM, f2=math.ceil(M / 32), f4=P(X), f5=ldx, f6=P(Xn), f7=ldn, f8=M, f9=K, f10=P(g), f11=P(b),
                   f12=fbits(eps))

    def zero(self, t):
        n = t.numel()
        assert t.dtype == torch.float32 and n % 4 == 0
        return rec(f0=OP_ZERO, f2=math.ceil(n / 4096), f4=P(t), f5=n)


# ---------------------------------------------------------------- weight export
def norm_of(m):
    m = unwrap(m)
    return (m.weight.detach().contiguous(), m.bias.detach().contiguous(), float(m.eps))


def tl_weights(layer):
    layer = unwrap(layer)
    a = layer.attn
    E = a.embed_dim
    f = layer.ffn
    w13 = torch.stack([f.w1.weight, f.w3.weight], dim=1).reshape(-1, E).contiguous()
    return dict(E=E, H=a.num_heads, F=f.w1.weight.shape[0], in_w=a.in_proj_weight.detach().contiguous(),
                in_b=a.in_proj_bias.detach().contiguous(), out_w=a.out_proj.weight.detach().contiguous(),
                out_b=a.out_proj.bias.detach().contiguous(), n1=norm_of(layer.attn_norm), n2=norm_of(layer.ffn_norm),
                w13=w13, w2=f.w2.weight.detach().contiguous())


def ffn_weights(seq):
    seq = unwrap(seq)
    norm, f = seq[0], seq[1]
    E = f.w1.weight.shape[1]
    w13 = torch.stack([f.w1.weight, f.w3.weight], dim=1).reshape(-1, E).contiguous()
    return dict(F=f.w1.weight.shape[0], n=norm_of(norm), w13=w13, w2=f.w2.weight.detach().contiguous())


def proj_weights(seq):
    seq = unwrap(seq)
    return dict(n=norm_of(seq[0]), w=seq[1].weight.detach().contiguous(), b=seq[1].bias.detach().contiguous())


def gdn_weights(g):
    g = unwrap(g)
    ws = [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight, g.g_proj.weight, g.a_proj.weight, g.b_proj.weight]
    wp = torch.cat(ws, 0)
    pad = (-wp.shape[0]) % 16
    wp = torch.cat([wp, wp.new_zeros(pad, wp.shape[1])], 0).contiguous()
    offs = np.cumsum([0] + [w.shape[0] for w in ws])
    assert g.num_heads == 6 and g.head_k_dim == 32 and g.head_v_dim == 64 and g.conv_size == 4
    return dict(wp=wp, ldp=wp.shape[0], q=int(offs[0]), k=int(offs[1]), v=int(offs[2]), g=int(offs[3]), a=int(offs[4]),
                b=int(offs[5]), cq=g.q_conv1d.weight.detach().reshape(-1, 4).contiguous(),
                ck=g.k_conv1d.weight.detach().reshape(-1, 4).contiguous(),
                cv=g.v_conv1d.weight.detach().reshape(-1, 4).contiguous(), A_log=g.A_log.detach().contiguous(),
                dt_bias=g.dt_bias.detach().contiguous(), onw=g.o_norm.weight.detach().contiguous(), oeps=float(g.o_norm.eps),
                wo=g.o_proj.weight.detach().contiguous())


def csr(groups_idx, targets_idx, n_groups, max_deg=None):
    """Return (ptr, idx) int32 CUDA tensors: for each group g, the list of targets."""
    gi = groups_idx.cpu().numpy()
    ti = targets_idx.cpu().numpy()
    order = np.argsort(gi, kind="stable")
    counts = np.bincount(gi, minlength=n_groups)
    assert max_deg is None or counts.max() <= max_deg
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return torch.tensor(ptr, device="cuda"), torch.tensor(ti[order].astype(np.int32), device="cuda")


class Model:
    def __init__(self, dec):
        self.dec = dec
        tg = dec.tanner_graph
        g0, gc, gr = tg[0], tg[...], tg[-1]
        pe = dec.global_pe().detach()
        self.n_init = int(dec.num_init_check)
        self.n_cyc = int(dec.num_cycle_check)
        self.n_data = int(dec.num_data_nodes)
        self.n_log = int(dec.num_logical_nodes)
        self.emb = dec.syndrome_embedding.weight.detach().contiguous()
        self.cpe0 = pe[g0.check_nodes].contiguous()
        self.dpe0 = pe[g0.data_nodes].contiguous()
        self.cpeC = pe[gc.check_nodes].contiguous()
        self.dpeC = pe[gc.data_nodes].contiguous()
        self.csr0 = csr(g0.data_to_check[0], g0.data_to_check[1], self.n_data, 16)
        self.csrC = csr(gc.data_to_check[0], gc.data_to_check[1], self.n_data, 16)
        self.csrR = csr(gr.data_to_check[0], gr.data_to_check[1], self.n_data, 16)
        self.csrL = csr(gc.data_to_logical[1], gc.data_to_logical[0], self.n_log)
        assert self.cpe0.shape[0] == self.n_init and self.cpeC.shape[0] == self.n_cyc
        assert self.dpe0.shape[0] == self.n_data == self.dpeC.shape[0]
        ce, re_ = unwrap(dec.cycle_encoder), unwrap(dec.readout_encoder)
        self.enc = {}
        for name, e in (("c", ce), ("r", re_)):
            self.enc[name] = dict(chk=[tl_weights(l) for l in e.check_transformer_layers],
                                  dat=[tl_weights(l) for l in e.data_transformer_layers],
                                  out=[tl_weights(l) for l in e.out_transformer_layers], proj=proj_weights(e.out_proj))
        d = unwrap(dec.decoder)
        self.dec_gdn = [gdn_weights(g) for g in d.recurrent_layers]
        self.dec_att = [tl_weights(l) for l in d.graph_attn_layers]
        self.dec_proj = proj_weights(d.out_proj)
        pm = dec.readout_pre_mixer
        self.cu_w = pm["cycle_update"].weight.detach().contiguous()
        self.cu_b = pm["cycle_update"].bias.detach().contiguous()
        self.ru_w = pm["readout_update"].weight.detach().contiguous()
        r = unwrap(dec.readout)
        self.ro_att = [tl_weights(l) for l in r.graph_attn_layers]
        self.ro_du = ffn_weights(r.data_update)
        self.ro_lu = ffn_weights(r.logical_update)
        self.ro_w = r.out_proj.weight.detach().reshape(-1).contiguous()
        self.ro_b = r.out_proj.bias.detach().contiguous()


# ---------------------------------------------------------------- program
def _al(n):
    return (n + 127) // 128 * 128


class Arena:
    """Flat bf16 scratch shared by program stages whose phase lifetimes do not overlap."""

    def __init__(self, prog, n_elems):
        self.n = int(n_elems)
        self.base = prog.buf(1, max(self.n, 128)).view(-1)

    def view(self, off, rows, cols):
        assert off % 128 == 0 and off + rows * cols <= self.n, (off, rows, cols, self.n)
        return self.base[off:off + rows * cols].view(rows, cols)


class Stream:
    def __init__(self, prog, X, M, groups, E, F, arena=None, off=0):
        self.X, self.M, self.groups, self.E, self.F = X, M, groups, E, F
        self.lat = Stream.is_lat(prog, M, E)
        self.end = off
        if self.lat:
            self.QKV = prog.buf(M, 3 * E)
            self.H = prog.buf(M, F)
            self.ACC = prog.buf(M, E, torch.float32)
            self.X1 = prog.buf(M, E)
        else:
            # throughput tlayer: QKV dies at attn before H is born at ffn_up; ATT dies at oproj before Xn is rewritten
            if arena is None:
                arena, off = Arena(prog, Stream.need(prog, M, E, F)), 0
            self.QKV = arena.view(off, M, 3 * E)
            self.H = arena.view(off, M, F)
            o2 = off + _al(M * max(3 * E, F))
            self.ATT = arena.view(o2, M, E)
            self.Xn = self.ATT
            self.end = o2 + _al(M * E)

    @staticmethod
    def is_lat(prog, M, E):
        return prog.fuse_attn and prog.can_res(M, E, True)

    @staticmethod
    def need(prog, M, E, F):
        return 0 if Stream.is_lat(prog, M, E) else _al(M * max(3 * E, F)) + _al(M * E)


def tlayer(prog, pairs, tag="tl"):
    """Latency streams: QKV(+zero acc) | attn+Wo (f32 atomics) | resadd+norm+FFN-up | FFN-down.
    Throughput streams: norm | QKV | attn (K/V once per group-head) | out-proj | norm | FFN-up | FFN-down."""
    any_thr = any(not s.lat for s, _ in pairs)
    ph = [[] for _ in range(7)]
    for s, w in pairs:
        E, F, H = w["E"], w["F"], w["H"]
        hd = E // H
        if s.lat:
            k = (1, 2, 5, 6) if any_thr else (0, 1, 2, 3)
            ph[k[0]].append(prog.gemm(s.M, 3 * E, E, s.X, E, w["in_w"], w["in_b"], s.QKV, 3 * E, EPI_STORE, norm=w["n1"]))
            ph[k[0]].append(prog.zero(s.ACC))
            for row0, gsz, ng in s.groups:
                ph[k[1]].append(prog.attn(s.QKV, E, hd, H, s.ACC, row0, gsz, ng, wo=w["out_w"]))
            ph[k[2]].append(prog.gemm(s.M, 2 * F, E, s.X, E, w["w13"], None, s.H, F, EPI_SWIGLU, norm=w["n2"],
                                      resadd=(s.ACC, E, w["out_b"], s.X1, E)))
            ph[k[3]].append(prog.gemm(s.M, E, F, s.H, F, w["w2"], None, s.X, E, EPI_RESID, aux1=s.X1, ldr=E))
        else:
            ph[0].append(prog.norm(s.X, E, s.Xn, E, s.M, E, w["n1"]))
            ph[1].append(prog.gemm(s.M, 3 * E, E, s.Xn, E, w["in_w"], w["in_b"], s.QKV, 3 * E, EPI_STORE, res=False))
            for row0, gsz, ng in s.groups:
                ph[2].append(prog.attn(s.QKV, E, hd, H, s.ATT, row0, gsz, ng, whole=True))
            ph[3].append(prog.gemm(s.M, E, E, s.ATT, E, w["out_w"], w["out_b"], s.X, E, EPI_RESID, aux1=s.X, ldr=E, res=False))
            ph[4].append(prog.norm(s.X, E, s.Xn, E, s.M, E, w["n2"]))
            ph[5].append(prog.gemm(s.M, 2 * F, E, s.Xn, E, w["w13"], None, s.H, F, EPI_SWIGLU, res=False))
            ph[6].append(prog.gemm(s.M, E, F, s.H, F, w["w2"], None, s.X, E, EPI_RESID, aux1=s.X, ldr=E, res=False))
    names = ("norm1", "qkv", "attn", "oproj", "norm2", "ffn_up", "ffn_dn")
    for p, name in zip(ph, names):
        prog.phase(p, f"{tag}.{name}")


def build_program(m: Model, L, **prog_kw):
    pr = Prog(**prog_kw)
    ni, nc, nd, nl = m.n_init, m.n_cyc, m.n_data, m.n_log
    n_det = 2 * ni + (L - 1) * nc
    Mc = ni + (L - 1) * nc
    E = 192
    XC, XD = pr.buf(Mc, E), pr.buf(L * nd, E)
    RC, RD = pr.buf(ni, E), pr.buf(nd, E)

    def embed(syn_off, stride, ng, nchk, cpe, dpe, cs, outC, outD):
        chunks = math.ceil((nchk + nd) / 16)
        return rec(f0=OP_EMBED, f2=ng * chunks, f4=syn_off, f5=stride, f6=ng, f7=nchk, f8=nd, f9=P(m.emb), f10=P(cpe),
                   f11=P(dpe), f12=P(cs[0]), f13=P(cs[1]), f14=P(outC), f15=P(outD), f16=E, f17=chunks, f18=E)

    def scat_mid(xc, cstride, xd, ng, cs):
        chunks = math.ceil(nd / 16)
        return rec(f0=OP_SCAT_MID, f2=ng * chunks, f4=P(xc), f5=cstride, f6=P(xd), f7=ng, f8=nd, f9=P(cs[0]),
                   f10=P(cs[1]), f11=E, f12=E, f13=chunks)

    ops = [embed(0, 0, 1, ni, m.cpe0, m.dpe0, m.csr0, XC, XD)]
    if L > 1:
        ops.append(embed(ni, nc, L - 1, nc, m.cpeC, m.dpeC, m.csrC, XC[ni:], XD[nd:]))
    ops.append(embed(n_det - ni, 0, 1, ni, m.cpe0, m.dpe0, m.csrR, RC, RD))
    pr.phase(ops)

    # encoder streams and decoder layers never coexist in a phase, so they share one scratch arena
    Md, D = L * nd, 256
    ldp = m.dec_gdn[0]["ldp"]
    assert all(g["ldp"] == ldp for g in m.dec_gdn)
    enc_need = Stream.need(pr, Mc, E, 576) + Stream.need(pr, Md, E, 576)
    dec_need = max(_al(Md * ldp) + _al(Md * 384), Stream.need(pr, Md, D, 768))
    SCR = Arena(pr, max(enc_need, dec_need))
    cgroups = [(0, ni, 1)] + ([(ni, nc, L - 1)] if L > 1 else [])
    s_cc = Stream(pr, XC, Mc, cgroups, E, 576, SCR, 0)
    s_cd = Stream(pr, XD, Md, [(0, nd, L)], E, 576, SCR, s_cc.end)
    s_rc = Stream(pr, RC, ni, [(0, ni, 1)], E, 576)
    s_rd = Stream(pr, RD, nd, [(0, nd, 1)], E, 576)
    ec, er = m.enc["c"], m.enc["r"]
    for i in range(len(ec["chk"])):
        tlayer(pr, [(s_cc, ec["chk"][i]), (s_cd, ec["dat"][i]), (s_rc, er["chk"][i]), (s_rd, er["dat"][i])], "enc_a")
    ops = [scat_mid(XC, ni, XD, 1, m.csr0)]
    if L > 1:
        ops.append(scat_mid(XC[ni:], nc, XD[nd:], L - 1, m.csrC))
    ops.append(scat_mid(RC, ni, RD, 1, m.csrR))
    pr.phase(ops, "scat_mid")
    for i in range(len(ec["out"])):
        tlayer(pr, [(s_cd, ec["out"][i]), (s_rd, er["out"][i])], "enc_b")

    XDEC = pr.buf(L * nd, D)
    R = pr.buf(nd, E)
    pc, prj = ec["proj"], er["proj"]
    pr.phase([pr.gemm(L * nd, D, E, XD, E, pc["w"], pc["b"], XDEC, D, EPI_STORE, norm=pc["n"]),
              pr.gemm(nd, E, E, RD, E, prj["w"], prj["b"], R, E, EPI_STORE, norm=prj["n"])])

    T2 = pr.buf(nd, E)
    last_rows = slice((L - 1) * nd, L * nd)
    nlayers = len(m.dec_gdn)
    for i in range(nlayers):
        g = m.dec_gdn[i]
        last = i == nlayers - 1
        Pb = SCR.view(0, Md, ldp)
        Ob = SCR.view(_al(Md * ldp), Md, 384)
        ops = [pr.gemm(L * nd, g["ldp"], D, XDEC, D, g["wp"], None, Pb, g["ldp"], EPI_STORE)]
        if i == 0:
            ops.append(pr.gemm(nd, E, E, R, E, m.ru_w, None, T2, E, EPI_STORE))
        pr.phase(ops, "dec.gdn_proj")
        n_pairs = nd * 6
        pr.phase([rec(f0=OP_GDN, f2=math.ceil(n_pairs / 8), f4=P(Pb), f5=g["ldp"], f6=g["q"], f7=g["k"], f8=g["v"],
                      f9=g["g"], f10=g["a"], f11=g["b"], f12=P(g["cq"]), f13=P(g["ck"]), f14=P(g["cv"]),
                      f15=P(g["A_log"]), f16=P(g["dt_bias"]), f17=P(g["onw"]), f18=P(Ob), f19=384, f20=nd, f21=L,
                      f22=6, f23=(L - 1) if last else 0, f24=fbits(g["oeps"]))], "dec.gdn_rec")
        if last:
            Xs, Ms, groups, Os = XDEC[last_rows], nd, [(0, nd, 1)], Ob[last_rows]
        else:
            Xs, Ms, groups, Os = XDEC, L * nd, [(0, nd, L)], Ob
        pr.phase([pr.gemm(Ms, D, 384, Os, 384, g["wo"], None, Xs, D, EPI_STORE)], "dec.gdn_oproj")
        # Pb/Ob are dead once gdn_oproj has run, so the attention stream reuses the arena from offset 0
        tlayer(pr, [(Stream(pr, Xs, Ms, groups, D, 768, SCR, 0), m.dec_att[i])], "dec")

    Dst = pr.buf(nd, E)
    dp = m.dec_proj
    pr.phase([pr.gemm(nd, E, D, XDEC[last_rows], D, dp["w"], dp["b"], Dst, E, EPI_STORE, norm=dp["n"])])
    V0 = pr.buf(nd, E)
    pr.phase([pr.gemm(nd, E, E, Dst, E, m.cu_w, m.cu_b, V0, E, EPI_MIX, aux1=T2, ldr=E, aux2=Dst, aux3=R)])
    s_ro = Stream(pr, V0, nd, [(0, nd, 1)], E, 576)
    for w in m.ro_att:
        tlayer(pr, [(s_ro, w)], "ro")
    du = m.ro_du
    Hd = pr.buf(nd, du["F"])
    pr.phase([pr.gemm(nd, 2 * du["F"], E, V0, E, du["w13"], None, Hd, du["F"], EPI_SWIGLU, norm=du["n"])])
    pr.phase([pr.gemm(nd, E, du["F"], Hd, du["F"], du["w2"], None, V0, E, EPI_RESID_TANH, aux1=V0, ldr=E)])
    L12 = pr.buf(nl, E)
    pr.phase([rec(f0=OP_SCAT_LOG, f2=1, f4=P(V0), f5=E, f6=nl, f7=P(m.csrL[0]), f8=P(m.csrL[1]), f9=P(L12), f10=E,
                  f11=nd)], "scat_log")
    lu = m.ro_lu
    Hl = pr.buf(nl, lu["F"])
    pr.phase([pr.gemm(nl, 2 * lu["F"], E, L12, E, lu["w13"], None, Hl, lu["F"], EPI_SWIGLU, norm=lu["n"])])
    pr.phase([pr.gemm(nl, E, lu["F"], Hl, lu["F"], lu["w2"], None, L12, E, EPI_RESID, aux1=L12, ldr=E)])
    pr.phase([rec(f0=OP_FINAL, f2=1, f4=P(L12), f5=E, f6=nl, f7=P(m.ro_w), f8=P(m.ro_b), f9=E)])
    pr.n_det = n_det
    pr.L = L
    pr.bufs = dict(XC=XC, XD=XD, RC=RC, RD=RD, XDEC=XDEC, R=R, T2=T2, D=Dst, V0=V0, L12=L12)
    return pr.finalize()


# ---------------------------------------------------------------- launcher
class MK:
    def __init__(self, lib_path, barrier_mode=1, prefetch=0):
        self.lib = ctypes.CDLL(str(lib_path))
        vp, ci, cll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        self.lib.mk_launch.argtypes = [vp, vp, vp, vp, ci, vp, ci, ci, ci, vp, ci, vp, vp, ci, vp, vp, ci, ci]
        self.lib.mk_e2e_prepare.argtypes = [vp, vp, vp, vp, ci, vp, cll, vp, ci, vp, vp, ci, vp, vp, ci, ci, vp, ci, ci, ci]
        self.lib.mk_e2e_call.argtypes = [ci]
        self.lib.mk_barrier_probe.argtypes = [vp, ci, vp, ci, vp, ci]
        self.lib.mk_occupancy.argtypes = [ctypes.POINTER(ci)]
        occ = ci(0)
        assert self.lib.mk_occupancy(ctypes.byref(occ)) == 0
        self.blocks_per_sm = occ.value
        self.n_sm = torch.cuda.get_device_properties(0).multi_processor_count
        self.bar = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.barrier_mode = barrier_mode
        self.prefetch = prefetch

    def run(self, prog, syn_u8, n_samples, n_phases=None, grid=None, nlog=12, phase_profile=False):
        logits = torch.zeros(n_samples, nlog, dtype=torch.float32, device="cuda")
        times = torch.zeros(2 * n_samples, dtype=torch.int64, device="cuda")
        pt = torch.zeros(prog.n_phases, dtype=torch.int64, device="cuda") if phase_profile else None
        grid = grid or self.n_sm * self.blocks_per_sm
        self.bar.zero_()
        err = self.lib.mk_launch(P(prog.ops_t), P(prog.pb_t), P(prog.tot_t), P(prog.end_t), n_phases or prog.n_phases,
                                 P(syn_u8), prog.n_det, syn_u8.shape[0], n_samples, P(logits), nlog, P(times),
                                 P(self.bar), grid, torch.cuda.current_stream().cuda_stream, P(pt), self.barrier_mode,
                                 self.prefetch)
        assert err == 0, f"launch error {err}"
        torch.cuda.synchronize()
        t = times.view(-1, 2).cpu().numpy()
        if phase_profile:
            ends = pt.cpu().numpy()
            starts = np.concatenate([[t[-1, 0]], ends[:-1]])
            self.last_phase_us = (ends - starts) / 1e3
        return logits, (t[:, 1] - t[:, 0]) / 1e3, t

    def run_prealloc(self, prog, syn_u8, logits, times, n_samples=1, n_phases=None,
                     grid=None, nlog=12):
        raise NotImplementedError("use E2E: the P3 probe allocated inside the timed loop")

    def barrier_cost_ns(self, iters=20000, grid=None, mode=None):
        t = torch.zeros(2, dtype=torch.int64, device="cuda")
        grid = grid or self.n_sm
        self.bar.zero_()
        mode = self.barrier_mode if mode is None else mode
        assert self.lib.mk_barrier_probe(P(self.bar), iters, P(t), grid, torch.cuda.current_stream().cuda_stream, mode) == 0
        torch.cuda.synchronize()
        tt = t.cpu().numpy()
        return (tt[1] - tt[0]) / iters


class E2E:
    """Preallocated single-sample host round trip over distinct pinned rows x_pinned[i].

    Every buffer (device syntax row, per-call logits/timer slots, per-call zeroed barrier counters, pinned
    logits) is allocated here, so call(i) performs no allocation and launches no auxiliary kernel.
    """

    BAR_STRIDE = 32  # int32 words: one 128-byte line per call

    def __init__(self, mk, prog, x_pinned, nlog=12):
        assert x_pinned.is_pinned() and x_pinned.dtype == torch.uint8 and x_pinned.is_contiguous()
        n, n_det = x_pinned.shape
        assert n_det == prog.n_det
        self.n, self.nlog, self.mk, self.prog, self.x = n, nlog, mk, prog, x_pinned
        self.d_syn = torch.zeros(n_det, dtype=torch.uint8, device="cuda")
        self.d_logits = torch.zeros(n, nlog, dtype=torch.float32, device="cuda")
        self.h_logits = torch.zeros(n, nlog, dtype=torch.float32, pin_memory=True)
        self.times = torch.zeros(n, 2, dtype=torch.int64, device="cuda")
        self.bars = torch.zeros(n, self.BAR_STRIDE, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        err = mk.lib.mk_e2e_prepare(P(prog.ops_t), P(prog.pb_t), P(prog.tot_t), P(prog.end_t), prog.n_phases,
                                    P(x_pinned), n_det, P(self.d_syn), n_det, P(self.d_logits), P(self.h_logits),
                                    nlog, P(self.times), P(self.bars), self.BAR_STRIDE, mk.n_sm * mk.blocks_per_sm,
                                    torch.cuda.current_stream().cuda_stream, mk.barrier_mode, mk.prefetch, n)
        assert err == 0, f"prepare error {err}"
        self.call = mk.lib.mk_e2e_call

    def device_ms(self):
        t = self.times.cpu().numpy()
        return (t[:, 1] - t[:, 0]) / 1e6

    def reset_slots(self, idx):
        """Re-zero barrier counters of already used slots (outside timing) so they can be called again."""
        self.bars[idx] = 0
        torch.cuda.synchronize()


def stock_logits(dec, inputs_bool, n):
    """Stock eager forward, batch size 1; returns logits in the model's native dtype."""
    out = []
    with torch.inference_mode():
        for i in range(n):
            raw = torch.tensor(inputs_bool[i:i + 1], dtype=torch.long, device="cuda")
            out.append(dec._simple_forward(split(dec, raw)))
    return torch.cat(out)


def decide(logits, dtype):
    """Public-decode decision rule: round(sigmoid(out)) evaluated in the model output dtype."""
    with torch.inference_mode():
        return torch.sigmoid(logits.to(dtype)).round().bool().cpu().numpy()


def compare(mk_logits, ref_logits):
    d = (mk_logits - ref_logits).abs()
    flips = ((mk_logits > 0) != (ref_logits > 0)).any(dim=1)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "flip_shots": int(flips.sum()),
            "n": int(mk_logits.shape[0]), "ref_abs_mean": float(ref_logits.abs().mean())}
