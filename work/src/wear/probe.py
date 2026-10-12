"""Hardware probe: how fast does the public window model (goodpjw2008 / jiweiliu Net, d=256, bs 512) train on this
machine's CPU? Used to decide whether a TPU-VM session (large CPU host) can replace a busy GPU session."""
import os, platform, subprocess, time


def main(cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("cpu_count", os.cpu_count(), "| platform", platform.platform(), flush=True)
    try:
        mem = [l for l in open("/proc/meminfo").read().splitlines() if l.startswith(("MemTotal", "MemAvailable"))]
        print(mem, flush=True)
        print(subprocess.run(["lscpu"], capture_output=True, text=True).stdout[:1500], flush=True)
    except Exception as e:
        print("meminfo/lscpu failed", e)
    print("torch", torch.__version__, "threads", torch.get_num_threads(), "mkldnn", torch.backends.mkldnn.is_available(),
          flush=True)
    for name in ("torch_xla", "lightgbm", "sklearn", "scipy", "pandas", "joblib", "numpy"):
        try:
            mod = __import__(name)
            print(name, getattr(mod, "__version__", "?"), flush=True)
        except Exception as e:
            print("missing", name, e, flush=True)

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, k, d=1):
            super().__init__()
            self.c = nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d)
            self.bn = nn.BatchNorm1d(cout)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            return F.gelu(self.bn(self.c(x))) + self.skip(x)

    class Net(nn.Module):
        def __init__(self, d=256, video=True):
            super().__init__()
            self.video = video
            self.sens = nn.Embedding(4, 32)
            self.imu = nn.Sequential(ConvBlock(4 + 32, 64, 7), ConvBlock(64, 128, 5), ConvBlock(128, 128, 5, 2),
                                     ConvBlock(128, d, 3, 4), ConvBlock(d, d, 3, 8))
            self.imu_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
            self.vin = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, d), nn.GELU(), nn.Dropout(0.2))
            self.vid_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.3))
            self.fuse = nn.Sequential(nn.Linear(2 * d + 32, d), nn.GELU(), nn.Dropout(0.3), nn.Linear(d, 19))

        def forward(self, a, v, s):
            e = self.sens(s)
            x = torch.cat([a.transpose(1, 2), e[:, :, None].expand(-1, -1, a.shape[1])], 1)
            h = self.imu(x)
            hi = self.imu_out(torch.cat([h.mean(-1), h.amax(-1)], 1))
            z = self.vin(v)
            hv = self.vid_out(torch.cat([z.mean(1), z.std(1)], 1))
            return self.fuse(torch.cat([hi, hv, e], 1))

    n_all = os.cpu_count() or 4
    for threads in sorted({n_all, max(n_all // 2, 1), min(16, n_all), 4}, reverse=True):
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        m = Net()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        a, v = torch.randn(512, 50, 4), torch.randn(512, 15, 768)
        s, y = torch.randint(0, 4, (512,)), torch.randint(0, 19, (512,))
        for i in range(25):
            if i == 5:
                t0 = time.time()
            loss = F.cross_entropy(m(a, v, s), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sps = 20 / (time.time() - t0)
        print(f"threads {threads}: {sps:.2f} train steps/s (bs 512) -> 18,000-step fusion run "
              f"{18000 / sps / 60:.0f} min", flush=True)
        with torch.no_grad():
            m.eval()
            t0 = time.time()
            for _ in range(10):
                m(a, v, s)
            print(f"   inference {10 * 512 / (time.time() - t0):.0f} samples/s", flush=True)
