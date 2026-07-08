#!/usr/bin/env python3
"""系统资源监控 — RK3588 CPU/内存/NPU 实时使用率"""

import os, time, threading, subprocess


def get_cpu_times():
    with open("/proc/stat") as f:
        line = f.readline().strip().split()
    vals = [int(v) for v in line[1:]]
    return sum(vals), vals[3]


def get_cpu_per_core():
    result = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and not line.startswith("cpu "):
                p = line.strip().split()
                v = [int(x) for x in p[1:]]
                result[p[0]] = (sum(v), v[3])
    return result


def get_memory():
    with open("/proc/meminfo") as f:
        data = f.read()
    total = int([l for l in data.split("\n") if "MemTotal" in l][0].split()[1]) / 1024
    avail = int([l for l in data.split("\n") if "MemAvailable" in l][0].split()[1]) / 1024
    return total, avail, (total - avail) / total * 100


def get_npu():
    try:
        r = subprocess.run(["cat", "/sys/class/devfreq/fdab0000.npu/cur_freq"], capture_output=True, text=True, timeout=1)
        if r.returncode == 0:
            freq = int(r.stdout.strip()) / 1000000
            return f"{freq:.0f}MHz{' (工作)' if freq > 500 else ' (闲置)'}"
    except:
        pass
    return "N/A"


def get_temp():
    try:
        r = subprocess.run(["cat", "/sys/class/thermal/thermal_zone0/temp"], capture_output=True, text=True, timeout=1)
        if r.returncode == 0:
            return int(r.stdout.strip()) / 1000
    except:
        pass
    return 0


class Monitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self._running = False

    def _update(self):
        nc = get_cpu_times()
        ncor = get_cpu_per_core()
        mt, ma, mp = get_memory()
        npu = get_npu()
        temp = get_temp()

        if not hasattr(self, "_pc"):
            self._pc = nc
            self._pcor = ncor
            return

        dt = nc[0] - self._pc[0]
        di = nc[1] - self._pc[1]
        cpu_total = (1 - di / max(dt, 1)) * 100

        core_pct = {}
        for c, (t, i) in ncor.items():
            pt, pi = self._pcor.get(c, (t, i))
            d_t = t - pt
            d_i = i - pi
            core_pct[c] = (1 - d_i / max(d_t, 1)) * 100

        self._pc = nc
        self._pcor = ncor

        # cpu0-3=A55, cpu4-7=A76 (实测 cpu_capacity)
        little = sum(core_pct.get(f"cpu{i}", 0) for i in range(4)) / 4
        big = sum(core_pct.get(f"cpu{i}", 0) for i in range(4, 8)) / 4

        lines = ["\033[H\033[J"]
        lines.append("===== RK3588 资源监控 =====")
        if temp:
            lines.append(f"温度: {temp:.1f}C")
        lines.append("")
        lines.append(f"CPU 总: {cpu_total:.0f}%")
        lines.append(f"  大核 A76: {big:.0f}%  小核 A55: {little:.0f}%")
        cl = "  "
        for i in range(8):
            cl += f"C{i}:{core_pct.get(f'cpu{i}',0):.0f}%  "
        lines.append(cl)
        lines.append("")
        lines.append(f"内存: {mp:.0f}% ({mt-ma:.0f}/{mt:.0f}MB)")
        lines.append(f"NPU: {npu}")
        print("\n".join(lines))

    def _loop(self):
        while self._running:
            self._update()
            time.sleep(self.interval)

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False


def main():
    import argparse
    p = argparse.ArgumentParser(description="RK3588 资源监控")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    mon = Monitor(interval=args.interval)
    try:
        mon.start()
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()
        print("\n停止")


if __name__ == "__main__":
    main()
