# gen_chart.py — publication-grade, spacious charts (vector PDF)
# Replace your current file with this version.

import os, json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RESULTS_DIR = r".\results"
OUT_DIR     = r".\Images"
PHASES      = ["early", "mid", "final"]

COLORS = {
    "ppo": "#0000ee",           # blue
    "random": "#5e6b7a",        # slate
    "static_soa": "#2aaa2a",    # green
    "static_micro": "#d62728",  # red
}
LABELS = {
    "ppo": "PPO",
    "random": "Random",
    "static_soa": "Static SOA",
    "static_micro": "Static Microservices",
}

BIN_SECONDS = 60
MAX_MINUTES = None

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def _rc_pub():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

def _save_pdf(name, legend_out=False):
    if legend_out:
        plt.gcf().subplots_adjust(right=0.74)  # extra room for legend
    plt.savefig(os.path.join(OUT_DIR, name), format="pdf", bbox_inches="tight")
    print("Saved:", os.path.join(OUT_DIR, name))
    plt.close()

def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_last_metrics(entry):
    li = entry.get("last_info") or entry.get("info") or {}
    m  = li.get("metrics", li)
    return m if isinstance(m, dict) else {}

def _collect_ppo_phase(phase):
    fp = os.path.join(RESULTS_DIR, f"{phase}_phase_test_log.json")
    if not os.path.exists(fp): return []
    data = _load_json(fp)
    out = []
    for ep in data:
        r = float(ep.get("total_reward", 0.0))
        m = _safe_last_metrics(ep)
        out.append((r, m))
    return out

def _collect_baseline_phase(phase, policy):
    fp = os.path.join(RESULTS_DIR, f"baseline_{phase}_{policy}.json")
    if not os.path.exists(fp): return []
    data = _load_json(fp)
    out = []
    for ep in data.get("episodes_detail", []):
        r = float(ep.get("total_reward", 0.0))
        m = _safe_last_metrics(ep)
        out.append((r, m))
    return out

def _metrics(items, key):
    vals = []
    for _, m in items:
        if isinstance(m, dict) and key in m and isinstance(m[key], (int, float)):
            vals.append(float(m[key]))
    return vals

def _choices(items):
    cs = []
    for _, m in items:
        c = m.get("choice")
        if isinstance(c, str): cs.append(c)
    return cs

def _ensure_headroom(ax, ratio=0.26):
    y0, y1 = ax.get_ylim()
    ymax = max([p.get_height() for p in ax.patches] + [y1])
    if ymax <= 0: ymax = y1
    ax.set_ylim(y0, ymax*(1+ratio))

def _annotate_bars(ax, *, percent=False, decimals=2, dy_frac=0.02,
                   rotation=0, inside=False):
    """Place value labels cleanly; can rotate or place inside bars."""
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for p in ax.patches:
        h = p.get_height()
        x = p.get_x() + p.get_width()/2
        txt = f"{h:.0f}%" if percent else f"{h:.{decimals}f}"
        if inside:
            y = h - span*0.035
            va = "top"; color = "white"
            bbox = dict(boxstyle="round,pad=0.15", facecolor="black",
                        alpha=0.25, edgecolor="none")
        else:
            y = h + span*dy_frac
            va = "bottom"; color = "black"; bbox = None
        ax.text(x, y, txt, ha="center", va=va, rotation=rotation, fontsize=11,
                color=color, bbox=bbox, clip_on=False, zorder=5)
        p.set_edgecolor("black"); p.set_linewidth(0.6)

# place radar axis labels OUTSIDE the circle using Axes coords (not clipped)
def _radar_labels_outside(ax, angles, labels, offset=0.075):
    # angles are in radians, zero at +x; our radar is rotated by +90°
    for ang, lab in zip(angles, labels):
        # convert to axes coords; center (0.5,0.5), radius ~0.5
        x = 0.5 + (0.5 + offset) * np.cos(ang - np.pi/2)
        y = 0.5 + (0.5 + offset) * np.sin(ang - np.pi/2)
        ha = "left" if np.cos(ang - np.pi/2) > 0.15 else ("right" if np.cos(ang - np.pi/2) < -0.15 else "center")
        ax.text(x, y, lab, transform=ax.transAxes, ha=ha, va="center", fontsize=13)

# ---------- load ----------
_rc_pub()
ppo = {ph: _collect_ppo_phase(ph) for ph in PHASES}
baselines = {
    ph: {
        "random": _collect_baseline_phase(ph, "random"),
        "static_soa": _collect_baseline_phase(ph, "static_soa"),
        "static_micro": _collect_baseline_phase(ph, "static_micro"),
    } for ph in PHASES
}

# ---------- charts ----------
def chart_ppo_reward_box():
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    data = [[r for r,_ in ppo[ph]] for ph in PHASES]
    ax.boxplot(data, tick_labels=[ph.capitalize() for ph in PHASES])
    ax.set_title("PPO Episode Reward Distribution (Testing)", pad=14)
    ax.set_ylabel("Episode Reward")
    ax.grid(True, axis="y", alpha=0.25)
    _save_pdf("ppo_reward_distribution_box.pdf")

def chart_avg_reward_grouped():
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    width = 0.24; x = np.arange(len(PHASES))
    def stats(v): return (float(np.mean(v)), float(np.std(v))) if v else (0.0, 0.0)
    series = {}
    for key in ["ppo","random","static_soa","static_micro"]:
        means, stds = [], []
        for ph in PHASES:
            arr = [r for r,_ in (ppo[ph] if key=="ppo" else baselines[ph][key])]
            m,s = stats(arr); means.append(m); stds.append(s)
        series[key]=(np.array(means), np.array(stds))
    ax.bar(x-1.5*width, series["ppo"][0], width, yerr=series["ppo"][1],
           color=COLORS["ppo"], label=LABELS["ppo"])
    ax.bar(x-0.5*width, series["random"][0], width, yerr=series["random"][1],
           color=COLORS["random"], label=LABELS["random"])
    ax.bar(x+0.5*width, series["static_soa"][0], width, yerr=series["static_oa"][1] if False else series["static_soa"][1],
           color=COLORS["static_soa"], label=LABELS["static_soa"])
    ax.bar(x+1.5*width, series["static_micro"][0], width, yerr=series["static_micro"][1],
           color=COLORS["static_micro"], label=LABELS["static_micro"])
    ax.set_xticks(x, [ph.capitalize() for ph in PHASES])
    ax.set_ylabel("Mean Episode Reward (±1σ)")
    ax.set_title("Average Reward: PPO vs Baselines", pad=16)
    ax.grid(True, axis="y", alpha=0.25)
    _ensure_headroom(ax, ratio=0.30)           # more headroom
    _annotate_bars(ax, percent=False, decimals=2, dy_frac=0.03)  # more offset
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_pdf("avg_reward_grouped.pdf", legend_out=True)

def chart_ppo_action_freq():
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    soa_pct, mic_pct = [], []
    for ph in PHASES:
        c = Counter(_choices(ppo[ph])); tot = sum(c.values()) or 1
        soa_pct.append(100*c.get("SOA",0)/tot)
        mic_pct.append(100*c.get("Microservices",0)/tot)
    x = np.arange(len(PHASES))
    ax.bar(x, soa_pct, color=COLORS["static_soa"], label="SOA")
    ax.bar(x, mic_pct, bottom=soa_pct, color=COLORS["static_micro"], label="Microservices")
    ax.set_xticks(x, [ph.capitalize() for ph in PHASES])
    ax.set_ylabel("Action Share (%)"); ax.set_ylim(0, 100)
    ax.set_title("PPO Action Frequency per Phase", pad=16)
    ax.grid(True, axis="y", alpha=0.25)
    for i,(s,m) in enumerate(zip(soa_pct, mic_pct)):
        if s>0: ax.text(i, s/2, f"{s:.0f}%", ha="center", va="center", fontsize=11)
        if m>0: ax.text(i, s+m/2, f"{m:.0f}%", ha="center", va="center", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_pdf("ppo_action_frequency_stacked.pdf", legend_out=True)

def chart_action_freq_compare():
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    width = 0.24; x = np.arange(len(PHASES))
    def micro(items):
        c = Counter(_choices(items)); tot = sum(c.values()) or 1
        return 100*c.get("Microservices",0)/tot
    ppo_m = [micro(ppo[ph]) for ph in PHASES]
    rnd_m = [micro(baselines[ph]["random"]) for ph in PHASES]
    soa_m = [micro(baselines[ph]["static_soa"]) for ph in PHASES]
    mic_m = [micro(baselines[ph]["static_micro"]) for ph in PHASES]
    ax.bar(x-1.5*width, ppo_m, width, color=COLORS["ppo"], label=LABELS["ppo"])
    ax.bar(x-0.5*width, rnd_m, width, color=COLORS["random"], label=LABELS["random"])
    ax.bar(x+0.5*width, soa_m, width, color=COLORS["static_soa"], label=LABELS["static_soa"])
    ax.bar(x+1.5*width, mic_m, width, color=COLORS["static_micro"], label=LABELS["static_micro"])
    ax.set_xticks(x, [ph.capitalize() for ph in PHASES])
    ax.set_ylim(0, 100); ax.set_ylabel("Microservices Action Share (%)")
    ax.set_title("Action Preference: PPO vs Baselines", pad=16)
    ax.grid(True, axis="y", alpha=0.25)
    _ensure_headroom(ax, ratio=0.12)
    _annotate_bars(ax, percent=True, decimals=0, dy_frac=0.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_pdf("action_freq_comparison.pdf", legend_out=True)

def _metric_grouped(metric_key, title, outname, ylabel, rotate_values=False):
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    width = 0.24; x = np.arange(len(PHASES))
    def avg(items):
        v = _metrics(items, metric_key)
        return float(np.mean(v)) if v else 0.0
    series = {
        "ppo":          [avg(ppo[ph]) for ph in PHASES],
        "random":       [avg(baselines[ph]["random"]) for ph in PHASES],
        "static_soa":   [avg(baselines[ph]["static_soa"]) for ph in PHASES],
        "static_micro": [avg(baselines[ph]["static_micro"]) for ph in PHASES],
    }
    for sh, key in zip([-1.5,-0.5,0.5,1.5], ["ppo","random","static_soa","static_micro"]):
        ax.bar(x+sh*width, series[key], width, color=COLORS[key], label=LABELS[key])
    ax.set_xticks(x, [ph.capitalize() for ph in PHASES])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.grid(True, axis="y", alpha=0.25)
    _ensure_headroom(ax, ratio=0.24)           # more space for labels
    _annotate_bars(ax, percent=False, decimals=2,
                   dy_frac=0.022, rotation=(90 if rotate_values else 0))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_pdf(outname, legend_out=True)

def chart_latency_energy_throughput():
    # rotate latency numbers 90° per your request to avoid any crowding
    _metric_grouped("latency",     "Average Latency (lower is better)",     "avg_latency_grouped.pdf",     "Latency", rotate_values=True)
    _metric_grouped("energy",      "Average Energy (lower is better)",      "avg_energy_grouped.pdf",      "Energy")
    _metric_grouped("throughput",  "Average Throughput (higher is better)", "avg_throughput_grouped.pdf",  "Throughput")

def chart_radar_per_phase():
    for ph in PHASES:
        rewards = [r for r,_ in ppo[ph]]
        if not rewards: continue
        lat = _metrics(ppo[ph], "latency"); en  = _metrics(ppo[ph], "energy"); th = _metrics(ppo[ph], "throughput")
        ppo_vec = np.array([np.mean(rewards), -np.mean(lat or [0]), -np.mean(en or [0]), np.mean(th or [0])])
        base = {}
        for pol in ["random","static_soa","static_micro"]:
            rr = [r for r,_ in baselines[ph][pol]]
            if not rr: continue
            latb = _metrics(baselines[ph][pol], "latency")
            enb  = _metrics(baselines[ph][pol], "energy")
            thb  = _metrics(baselines[ph][pol], "throughput")
            base[pol] = np.array([np.mean(rr), -np.mean(latb or [0]), -np.mean(enb or [0]), np.mean(thb or [0])])
        if not base: continue
        best = max(base, key=lambda k: base[k][0])
        base_vec = base[best]
        stack = np.vstack([ppo_vec, base_vec]); mn = stack.min(axis=0); mx = stack.max(axis=0)
        rng = np.where(mx-mn==0, 1, mx-mn)
        ppo_n  = (ppo_vec - mn)/rng
        base_n = (base_vec - mn)/rng

        labels = ["Reward", "Latency (inv)", "Energy (inv)", "Throughput"]
        ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        ang_closed = np.r_[ang, ang[0]]

        fig = plt.figure(figsize=(11.0, 8.6))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
        # plot first, then place custom labels OUTSIDE
        ax.set_rlabel_position(90)
        ax.set_ylim(0, 1.0)

        ax.plot(ang_closed, np.r_[base_n, base_n[0]], color=COLORS[best], linewidth=2, label=LABELS[best])
        ax.fill(ang_closed, np.r_[base_n, base_n[0]], color=COLORS[best], alpha=0.20)
        ax.plot(ang_closed, np.r_[ppo_n, ppo_n[0]],  color=COLORS["ppo"], linewidth=2, label=LABELS["ppo"])
        ax.fill(ang_closed, np.r_[ppo_n, ppo_n[0]],  color=COLORS["ppo"], alpha=0.25)

        # remove default theta labels & place clean ones outside the circle
        ax.set_thetagrids([], [])  # hide defaults
        _radar_labels_outside(ax, ang, labels, offset=0.095)

        plt.title(f"{ph.capitalize()} — Radar (normalized)", pad=16)
        plt.legend(loc="center left", bbox_to_anchor=(1.30, 0.5), frameon=False)
        _save_pdf(f"radar_{ph}.pdf", legend_out=True)

def chart_cumulative_reward():
    rewards_all = []
    for ph in PHASES: rewards_all += [r for r,_ in ppo[ph]]
    if not rewards_all: return
    cum = np.cumsum(rewards_all)
    n = len(cum)
    if n > 1000:
        idx = np.linspace(0, n-1, 1000).astype(int)
        cum = cum[idx]; xs = np.arange(1, len(cum)+1)
    else:
        xs = np.arange(1, n+1)
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    ax.plot(xs, cum, color=COLORS["ppo"], linewidth=2)
    ax.set_title("Cumulative Reward across PPO Test Episodes", pad=14)
    ax.set_xlabel("Episode Index (Early→Mid→Final)")
    ax.set_ylabel("Cumulative Reward")
    ax.grid(True, alpha=0.25)
    _save_pdf("ppo_cumulative_reward.pdf")

def _load_actions_binned(bin_seconds=BIN_SECONDS):
    path = os.path.join(RESULTS_DIR, "step_logs.jsonl")
    if not os.path.exists(path): return None
    t0 = None; bins = {"SOA": Counter(), "Microservices": Counter()}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: obj = json.loads(line)
            except Exception: continue
            ch = obj.get("metrics", {}).get("choice") or obj.get("choice")
            ts = obj.get("ts")
            if ch not in ("SOA","Microservices"): continue
            if not isinstance(ts, (int,float)):
                ts = (t0 if t0 is not None else 0.0) + bin_seconds
            if t0 is None: t0 = ts
            idx = int((ts - t0)//bin_seconds)
            bins[ch][idx] += 1
    if t0 is None: return None
    maxi = max([max(c.keys()) if c else 0 for c in bins.values()])
    xs = np.arange(maxi+1)
    soa = np.array([bins["SOA"].get(i,0) for i in xs])
    mic = np.array([bins["Microservices"].get(i,0) for i in xs])
    tm = xs*bin_seconds/60.0
    if MAX_MINUTES is not None:
        mask = tm <= MAX_MINUTES; tm = tm[mask]; soa = soa[mask]; mic = mic[mask]
    return tm, soa, mic

def chart_action_rate_binned():
    res = _load_actions_binned()
    if res is None: return
    t, soa, mic = res
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    ax.plot(t, soa, label="SOA", color=COLORS["static_soa"], linewidth=2)
    ax.plot(t, mic, label="Microservices", color=COLORS["static_micro"], linewidth=2)
    ax.set_xlabel(f"Time (minutes), bin = {BIN_SECONDS}s")
    ax.set_ylabel("Actions per bin")
    ax.set_title("Action Rate Over Time (binned)", pad=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_pdf("action_rate_binned.pdf", legend_out=True)

def chart_action_heatmap_binned():
    res = _load_actions_binned()
    if res is None: return
    t, soa, mic = res
    soa_norm = soa / (soa.max() if soa.max()>0 else 1)
    mic_norm = mic / (mic.max() if mic.max()>0 else 1)
    Z = np.vstack([soa_norm, mic_norm])
    fig, ax = plt.subplots(figsize=(11.6, 4.0))
    extent=[t.min() if len(t)>0 else 0, t.max() if len(t)>0 else 1, -0.5, 1.5]
    Z1 = Z.copy(); Z1[1,:] = np.nan
    im1 = ax.imshow(Z1, aspect="auto", interpolation="nearest",
                    extent=extent, cmap="Greens", vmin=0, vmax=1)
    Z2 = Z.copy(); Z2[0,:] = np.nan
    im2 = ax.imshow(Z2, aspect="auto", interpolation="nearest",
                    extent=extent, cmap="Reds", vmin=0, vmax=1, alpha=0.9)
    ax.set_yticks([0,1]); ax.set_yticklabels(["SOA","Microservices"])
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Action Density (binned heatmap)", pad=10)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="2.5%", pad=0.2)
    cax2 = divider.append_axes("right", size="2.5%", pad=0.6)
    cb1 = plt.colorbar(im1, cax=cax1); cb1.set_label("SOA density")
    cb2 = plt.colorbar(im2, cax=cax2); cb2.set_label("Microservices density")
    _save_pdf("action_heatmap_binned.pdf")

def main():
    chart_ppo_reward_box()
    chart_avg_reward_grouped()
    chart_ppo_action_freq()
    chart_action_freq_compare()
    chart_latency_energy_throughput()
    chart_radar_per_phase()          # labels outside circle
    chart_cumulative_reward()
    chart_action_rate_binned()
    chart_action_heatmap_binned()
    print("\nAll charts saved to:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
