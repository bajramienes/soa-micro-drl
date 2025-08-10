import os
import time
import json
import numpy as np
from typing import Dict, Any

# Use the same env as PPO run
from drl_env import OrchestrationEnv

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Match your PPO test size so charts are 1:1 comparable
TEST_EPISODES_PER_PHASE = 50

# Policies to compare (no learning)
POLICIES = ["random", "static_soa", "static_micro"]

PHASES = [
    "Early Phase",
    "Mid Phase",
    "Final Phase",
]

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def choose_action(env: OrchestrationEnv, policy: str) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "static_soa":
        return 0
    if policy == "static_micro":
        return 1
    raise ValueError(f"Unknown policy: {policy}")

def run_baseline_phase(phase_name: str, policy: str) -> Dict[str, Any]:
    print(f"\n=== BASELINE {policy.upper()} — {phase_name} START ===")
    start_human = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_human}")
    t0 = time.time()

    env = OrchestrationEnv()  # plain Gymnasium env (not vec)
    episode_rewards = []
    ep_logs = []

    for ep in range(1, TEST_EPISODES_PER_PHASE + 1):
        obs, info = env.reset()
        total_r = 0.0
        terminated = False
        truncated = False
        step_infos = []

        while not (terminated or truncated):
            a = choose_action(env, policy)
            obs, reward, terminated, truncated, info = env.step(a)
            total_r += float(reward)
            step_infos.append(info)

        episode_rewards.append(total_r)
        ep_logs.append({
            "episode": ep,
            "total_reward": total_r,
            "last_info": step_infos[-1] if step_infos else {}
        })

        if ep % 10 == 0:
            print(f"[BASELINE {policy}] {phase_name}: episode {ep}/{TEST_EPISODES_PER_PHASE} "
                  f"(elapsed {(time.time()-t0)/60:.1f} min)")

    avg_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    dt = time.time() - t0
    end_human = time.strftime("%Y-%m-%d %H:%M:%S")

    out = {
        "phase": phase_name,
        "policy": policy,
        "episodes": TEST_EPISODES_PER_PHASE,
        "avg_reward": avg_r,
        "duration_min": round(dt/60, 2),
        "start_time": start_human,
        "end_time": end_human,
        "episodes_detail": ep_logs
    }

    # Save JSON
    fname = f"baseline_{phase_name.split()[0].lower()}_{policy}.json".replace(" ", "_")
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w") as f:
        json.dump(to_serializable(out), f, indent=2)
    print(f"[INFO] Saved {path}")
    print(f"=== BASELINE {policy.upper()} — {phase_name} END (duration {out['duration_min']} min) ===")
    return out

def main():
    summary = {"phases": [], "policies": POLICIES, "episodes_per_phase": TEST_EPISODES_PER_PHASE}
    grand_t0 = time.time()

    for phase in PHASES:
        for pol in POLICIES:
            result = run_baseline_phase(phase, pol)
            summary["phases"].append({
                "phase": phase,
                "policy": pol,
                "avg_reward": result["avg_reward"],
                "duration_min": result["duration_min"]
            })

    summary["total_duration_min"] = round((time.time()-grand_t0)/60, 2)
    with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as f:
        json.dump(to_serializable(summary), f, indent=2)

    print("\n=== BASELINE SUMMARY ===")
    for row in summary["phases"]:
        print(f"{row['phase']:<12} | {row['policy']:<12} | avg_reward={row['avg_reward']:.3f} | {row['duration_min']:.2f} min")
    print(f"TOTAL baseline time: {summary['total_duration_min']} min")

if __name__ == "__main__":
    main()
