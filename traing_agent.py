import os
import time
import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from drl_env import OrchestrationEnv


# === CONFIG ===
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TS_TOTAL_PER_PHASE = 60000
N_STEPS_PER_ITER   = TS_TOTAL_PER_PHASE
BATCH_SIZE         = 1500   # divides 60000 exactly
TEST_EPISODES      = 50
# ===============


def to_serializable(obj):
    """Recursively convert NumPy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _episode_rollout_vec(env, model):
    obs = env.reset()
    done = False
    total_r = 0.0
    infos_accum = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_r += float(np.asarray(rewards).item())
        done = bool(np.asarray(dones).item())
        infos_accum.append(infos[0] if isinstance(infos, (list, tuple)) and len(infos) else {})
    return total_r, infos_accum


def run_phase(model, env, phase_name, save_name):
    print(f"\n=== {phase_name} START ===")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()

    # TRAIN
    print(f"[TRAIN] {phase_name}: PPO.learn for {TS_TOTAL_PER_PHASE} timesteps ...")
    model.learn(total_timesteps=int(TS_TOTAL_PER_PHASE), reset_num_timesteps=False)
    model_path = os.path.join(RESULTS_DIR, save_name)
    model.save(model_path)
    print(f"[INFO] {phase_name} model saved to {model_path}.zip")

    # TEST
    print(f"[TEST]  {phase_name}: {TEST_EPISODES} deterministic episodes ...")
    test_logs = []
    te0 = time.time()
    for i in range(1, TEST_EPISODES + 1):
        total_r, infos = _episode_rollout_vec(env, model)
        test_logs.append({
            "episode": i,
            "total_reward": total_r,
            "last_info": infos[-1] if infos else {}
        })
        if i % 10 == 0:
            print(f"[TEST]  {phase_name}: episode {i}/{TEST_EPISODES} "
                  f"(elapsed {(time.time()-te0)/60:.1f} min)")

    # Convert all logs to JSON-serializable format
    test_logs = to_serializable(test_logs)

    # Save log
    out_json = os.path.join(RESULTS_DIR, f"{phase_name.lower().replace(' ', '_')}_test_log.json")
    with open(out_json, "w") as f:
        json.dump(test_logs, f, indent=2)
    print(f"[INFO] {phase_name} test log saved to {out_json}")

    dt = time.time() - t0
    print(f"[INFO] {phase_name} completed in {dt/60:.2f} min")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=== {phase_name} END ===")
    return dt


def train_all_phases():
    env = DummyVecEnv([lambda: OrchestrationEnv()])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=N_STEPS_PER_ITER,
        batch_size=BATCH_SIZE,
        tensorboard_log="./tensorboard_logs"
    )

    durations = []
    durations.append(("Early Phase", run_phase(model, env, "Early Phase", "ppo_early")))
    durations.append(("Mid Phase",   run_phase(model, env, "Mid Phase",   "ppo_mid")))
    durations.append(("Final Phase", run_phase(model, env, "Final Phase", "ppo_final")))

    print("\n=== SUMMARY (minutes) ===")
    for name, secs in durations:
        print(f"{name:<12}: {secs/60:.2f}")


if __name__ == "__main__":
    train_all_phases()
