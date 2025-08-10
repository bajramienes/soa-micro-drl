import os
import time
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OrchestrationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.max_steps = 50
        self.cur_step = 0
        self.state = self._sample_obs()

        # NEW: per-step sleep to control wall-clock runtime
        self.step_sleep = 0.06  # ~1 hour per phase with 60,000 steps

        os.makedirs("./results", exist_ok=True)
        self._step_log_path = "./results/step_logs.jsonl"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_step = 0
        self.state = self._sample_obs()
        return self.state, {}

    def step(self, action):
        action = int(action)
        # NEW: slow down to hit wall-clock target
        time.sleep(self.step_sleep)

        metrics = self._simulate_metrics(action)
        reward = float((metrics["throughput"] / max(metrics["latency"], 1e-6)) - (metrics["energy"] / 1000.0))

        self.cur_step += 1
        terminated = self.cur_step >= self.max_steps
        truncated = False

        self.state = self._sample_obs()
        info = {"metrics": metrics, "action": action, "step": int(self.cur_step), "ts": float(time.time())}

        try:
            with open(self._step_log_path, "a") as f:
                json.dump({"ts": info["ts"], "action": action, "step": int(self.cur_step), "metrics": metrics}, f)
                f.write("\n")
        except Exception:
            pass

        return self.state, reward, terminated, truncated, info

    def _sample_obs(self):
        return np.random.random(4).astype(np.float32)

    def _simulate_metrics(self, action: int):
        if action == 0:
            return {
                "latency": float(np.random.uniform(140, 220)),
                "energy": float(np.random.uniform(380, 560)),
                "throughput": float(np.random.uniform(50, 80)),
                "choice": "SOA",
            }
        else:
            return {
                "latency": float(np.random.uniform(80, 150)),
                "energy": float(np.random.uniform(300, 500)),
                "throughput": float(np.random.uniform(70, 110)),
                "choice": "Microservices",
            }
