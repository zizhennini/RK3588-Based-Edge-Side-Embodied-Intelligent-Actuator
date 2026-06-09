#!/usr/bin/env python3
"""监控 RK3588 资源占用 — 每核心 CPU / 内存 / NPU / 温度"""
import subprocess
import time


def get_cpu_per_core():
    """读取 /proc/stat 计算每核心占用率"""
    with open("/proc/stat") as f:
        lines = f.readlines()

    stats = {}
    for line in lines:
        if line.startswith("cpu"):
            parts = line.split()
            name = parts[0]
            if name == "cpu":
                continue
            vals = [int(v) for v in parts[1:]]
            total = sum(vals)
            idle = vals[3]
            stats[name] = (total, idle)
    return stats


def monitor(duration: int = 30, interval: float = 1.0):
    print(f"监控系统资源 {duration}s（每 {interval}s 刷新）")
    print()

    prev = get_cpu_per_core()
    time.sleep(0.2)

    start = time.time()
    while time.time() - start < duration:
        cur = get_cpu_per_core()

        per_core = []
        for core in sorted(cur.keys(), key=lambda x: int(x[3:])):
            if core in prev:
                p_total, p_idle = prev[core]
                c_total, c_idle = cur[core]
                d_total = c_total - p_total
                d_idle = c_idle - p_idle
                usage = (1 - d_idle / d_total) * 100 if d_total > 0 else 0
                per_core.append(usage)

        mem = subprocess.run(
            "free -m | awk 'NR==2{printf \"%.1f%%\", $3/$2*100}'",
            shell=True, capture_output=True, text=True
        ).stdout.strip()

        npu = subprocess.run(
            "cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo 'N/A'",
            shell=True, capture_output=True, text=True
        ).stdout.strip()

        temp = subprocess.run(
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf \"%.1f°C\", $1/1000}' || echo 'N/A'",
            shell=True, capture_output=True, text=True
        ).stdout.strip()

        elapsed = int(time.time() - start)
        core_str = " | ".join(f"{u:5.0f}%" for u in per_core)

        if elapsed == 0:
            print(f"{'核':>6} | {'A55':>5} {'A55':>5} {'A55':>5} {'A55':>5} | {'A76':>5} {'A76':>5} {'A76':>5} {'A76':>5} | {'MEM':>7} | {'NPU':>5} | {'TEMP':>7}")
            print("-" * 80)

        print(f"{elapsed:>5}s | {core_str} | {mem:>7} | {npu:>5} | {temp:>7}")

        prev = cur
        time.sleep(interval)


if __name__ == "__main__":
    monitor()
