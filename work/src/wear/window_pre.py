"""Window classifier (Briano's IMU 1D-CNN + pooled video MLP) - out-of-fold and test inference with the fitted
5-fold models published in jiweiliu/wear-timeline-graph-models (fold f model never saw the subjects of fold f)."""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import N_CLS, VLEN, WIN, blend, find_file, mean_logp, macro_f1, CFG

FUSION_RUNS = ("final_s0", "final_s1", "final_s2", "final_s3")
IMU_RUNS = ("imu_s0", "imu_s1")


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, d=1):
        super().__init__()
        self.c = nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d)
        self.bn = nn.BatchNorm1d(cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.gelu(self.bn(self.c(x))) + self.skip(x)


class Net(nn.Module):
    def __init__(self, d=256, vdrop=0.5):
        super().__init__()
        self.sens = nn.Embedding(4, 32)
        self.imu = nn.Sequential(ConvBlock(4 + 32, 64, 7), ConvBlock(64, 128, 5), ConvBlock(128, 128, 5, 2),
                                 ConvBlock(128, d, 3, 4), ConvBlock(d, d, 3, 8))
        self.imu_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.vin = nn.Sequential(nn.Dropout(vdrop), nn.Linear(768, d), nn.GELU(), nn.Dropout(0.2))
        self.vid_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.3))
        self.fuse = nn.Sequential(nn.Linear(2 * d + 32, d), nn.GELU(), nn.Dropout(0.3), nn.Linear(d, N_CLS))
        self.h_imu = nn.Linear(d, N_CLS)
        self.h_vid = nn.Linear(d, N_CLS)

    def forward(self, a, v, sensor):
        e = self.sens(sensor)
        x = torch.cat([a.transpose(1, 2), e[:, :, None].expand(-1, -1, a.shape[1])], 1)
        h = self.imu(x)
        hi = self.imu_out(torch.cat([h.mean(-1), h.amax(-1)], 1))
        z = self.vin(v)
        hv = self.vid_out(torch.cat([z.mean(1), z.std(1)], 1))
        return self.fuse(torch.cat([hi, hv, e], 1)), self.h_imu(hi), self.h_vid(hv)


def acc_stats(x, ok):
    m, sd = np.zeros((4, 3)), np.ones((4, 3))
    for k in range(4):
        xs = x[ok[:, k], k].astype(np.float64)
        if len(xs) > 100:
            m[k], sd[k] = xs.mean(0), xs.std(0) + 1e-3
    return m, sd


def vid_stats(v):
    v = v.astype(np.float32)
    return v.mean(0), v.std(0) + 1e-3


def norm_stats(P, recs, test_acc, test_vid, test_sensor, test_sbj):
    """per-subject normalisation exactly as in training (--anorm subject --vnorm subject); test uses its own windows"""
    SB = sorted({r["sbj"] for r in recs})
    TS = sorted(set(test_sbj.tolist()))
    G = 22 + len(TS)
    AM, AS = np.zeros((G, 4, 3)), np.ones((G, 4, 3))
    VM, VS = np.zeros((G, 768)), np.ones((G, 768))
    for sb in SB:
        ii = [i for i, r in enumerate(recs) if r["sbj"] == sb]
        acc = np.concatenate([np.load(P.PROC / f"{recs[i]['stem']}_acc.npy") for i in ii]).reshape(-1, 4, 3)
        val = np.concatenate([np.load(P.PROC / f"{recs[i]['stem']}_valid.npy") for i in ii])
        AM[sb], AS[sb] = acc_stats(acc, val)
        VM[sb], VS[sb] = vid_stats(np.concatenate([np.load(P.PROC / f"{recs[i]['stem']}_vid.npy", mmap_mode="r")[::3]
                                                   for i in ii]))
        del acc, val
    for ti, t in enumerate(TS):
        m = test_sbj == t
        x = np.zeros((m.sum(), WIN, 4, 3), np.float32)
        ok = np.zeros((m.sum(), WIN, 4), bool)
        x[np.arange(m.sum()), :, test_sensor[m]] = test_acc[m]
        ok[np.arange(m.sum()), :, test_sensor[m]] = True
        AM[22 + ti], AS[22 + ti] = acc_stats(x.reshape(-1, 4, 3), ok.reshape(-1, 4))
        VM[22 + ti], VS[22 + ti] = vid_stats(np.asarray(test_vid[m]).reshape(-1, 768))
    test_grp = np.array([22 + TS.index(t) for t in test_sbj])
    return (AM, AS, VM, VS), test_grp


class Infer:
    def __init__(self, stats, dev):
        self.dev = dev
        self.AM, self.AS, self.VM, self.VS = (torch.tensor(x, dtype=torch.float32, device=dev) for x in stats)

    def prep(self, a_raw, v_raw, sensor, g, only):
        a_raw = a_raw.float()
        mag = a_raw.norm(dim=-1, keepdim=True) - 1.0
        a = (a_raw - self.AM[g, sensor][:, None]) / self.AS[g, sensor][:, None]
        a = torch.cat([a, mag], -1)
        v = (v_raw.float() - self.VM[g][:, None]) / self.VS[g][:, None]
        if only == "imu":
            v = torch.zeros_like(v)
        return a, v

    @torch.no_grad()
    def predict(self, model, acc, vid, sensor, g, only, bs=1024):
        model.eval()
        outs = []
        for i in range(0, len(sensor), bs):
            a = torch.from_numpy(np.asarray(acc[i:i + bs], np.float32)).to(self.dev)
            v = torch.from_numpy(np.asarray(vid[i:i + bs])).to(self.dev)
            s = torch.from_numpy(sensor[i:i + bs]).to(self.dev)
            gg = torch.from_numpy(g[i:i + bs]).to(self.dev)
            a, v = self.prep(a, v, s, gg, only)
            outs.append(F.log_softmax(model(a, v, s)[0].float(), 1).cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def emb(self, vid, g, bs=4096):
        out = []
        for i in range(0, len(g), bs):
            v = torch.from_numpy(np.asarray(vid[i:i + bs])).to(self.dev).float()
            gg = torch.from_numpy(g[i:i + bs]).to(self.dev)
            out.append(((v - self.VM[gg][:, None]) / self.VS[gg][:, None]).mean(1).cpu().numpy())
        return np.concatenate(out)


def run(P, recs, d, t, model_root=None):
    """fills d/t with logp, fusion_logp, imu_logp, emb (and per-run logps for later blending)"""
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_root is None:
        model_root = find_file("final_s0_model_f0.pt").parent
    print("window models from", model_root, "device", dev, flush=True)
    t_acc, t_vid = t["acc"](), t["vid"]()
    stats, test_grp = norm_stats(P, recs, t_acc, t_vid, t["sensor"], t["sbj"])
    inf = Infer(stats, dev)
    d_acc, d_vid = d["acc"](), d["vid"]()
    d_grp = d["sbj"].astype(np.int64)
    runs_oof, runs_test = {}, {}
    t0 = time.time()
    for run_name in FUSION_RUNS + IMU_RUNS:
        only = "imu" if run_name.startswith("imu") else "none"
        oof = np.zeros((len(d["y"]), N_CLS), np.float32)
        tests = []
        for fold in range(5):
            model = Net().to(dev)
            model.load_state_dict(torch.load(model_root / f"{run_name}_model_f{fold}.pt", map_location=dev,
                                             weights_only=True))
            ii = np.flatnonzero(d["fold"] == fold)
            oof[ii] = inf.predict(model, d_acc[ii], d_vid[ii], d["sensor"][ii], d_grp[ii], only)
            tests.append(inf.predict(model, t_acc, t_vid, t["sensor"], test_grp, only))
            del model
        runs_oof[run_name] = oof
        x = np.mean(tests, 0)   # mean of the 5 folds' log-softmax, renormalised as in Briano's W4
        runs_test[run_name] = x - np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)) - x.max(1, keepdims=True)
        print(f"{run_name}: OOF macro F1 {macro_f1(d['y'], oof.argmax(1)):.4f} [{time.time() - t0:.0f}s]", flush=True)
    fus = mean_logp([runs_oof[r] for r in FUSION_RUNS])
    imu = mean_logp([runs_oof[r] for r in IMU_RUNS])
    d.update(fusion_logp=fus, imu_logp=imu, logp=blend(fus, imu, CFG["imu_w"]), emb=inf.emb(d_vid, d_grp))
    fus_t = mean_logp([runs_test[r] for r in FUSION_RUNS])
    imu_t = mean_logp([runs_test[r] for r in IMU_RUNS])
    t.update(fusion_logp=fus_t, imu_logp=imu_t, logp=blend(fus_t, imu_t, CFG["imu_w"]), emb=inf.emb(t_vid, test_grp))
    print(f"window-level OOF macro F1: fusion {macro_f1(d['y'], fus.argmax(1)):.4f} imu {macro_f1(d['y'], imu.argmax(1)):.4f} "
          f"blend {macro_f1(d['y'], d['logp'].argmax(1)):.4f}  (Briano 4F+2I: 0.7194 / 0.5995 / 0.7253)", flush=True)
    return runs_oof, runs_test
