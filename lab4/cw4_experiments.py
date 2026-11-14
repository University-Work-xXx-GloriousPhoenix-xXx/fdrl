"""
КП4. Дослідження DQN (Stable-Baselines3) у середовищі Acrobot-v1.

Джерело реалізації: Stable-Baselines3 (DQN), https://github.com/DLR-RM/stable-baselines3
Базові гіперпараметри: RL Baselines3 Zoo, hyperparams/dqn.yml, запис Acrobot-v1
(https://github.com/DLR-RM/rl-baselines3-zoo; ті самі значення наведено в картці моделі
https://huggingface.co/sb3/dqn-Acrobot-v1).

Власні зміни/налаштування:
  * скорочено бюджет навчання до N_TIMESTEPS = 60 000 кроків (замість 100 000) через обмежені ресурси CPU;
  * додано журнал навчальних епізодів (Monitor) та періодичне детерміноване оцінювання (EvalCallback);
  * серії експериментів з одним зміненим гіперпараметром відносно базової конфігурації, по 3 seed;
  * фінальне оцінювання кожної моделі на 50 епізодах з фіксованими seed.

Запуск:  python cw4_experiments.py [--workers 2]
Результати: results/<config>_s<seed>/{monitor.csv, evaluations.npz, model.zip, final.json}
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

N_TIMESTEPS = 60_000
EVAL_FREQ = 3_000
N_EVAL_EPISODES = 10
N_FINAL_EPISODES = 50
SEEDS = [0, 1, 2]
ENV_ID = "Acrobot-v1"

BASE = dict(
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50_000,
    learning_starts=0,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
)

# Кожна конфігурація змінює рівно один гіперпараметр відносно BASE
CONFIGS = {
    "base":         {},
    "lr_1e-4":      dict(learning_rate=1e-4),
    "lr_3e-3":      dict(learning_rate=3e-3),
    "target_10":    dict(target_update_interval=10),
    "target_2000":  dict(target_update_interval=2000),
    "net_16":       dict(policy_kwargs=dict(net_arch=[16])),
    "net_64x64":    dict(policy_kwargs=dict(net_arch=[64, 64])),
    "explore_0.5":  dict(exploration_fraction=0.5),
    "eps_end_0.01": dict(exploration_final_eps=0.01),
}


def run_one(args):
    name, seed, out_root = args
    out = os.path.join(out_root, f"{name}_s{seed}")
    if os.path.exists(os.path.join(out, "final.json")):
        return name, seed, "cached"
    os.makedirs(out, exist_ok=True)

    import numpy as np
    import torch
    import gymnasium as gym
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback

    torch.set_num_threads(1)
    hp = {**BASE, **CONFIGS[name]}

    env = Monitor(gym.make(ENV_ID), filename=os.path.join(out, "monitor"))
    eval_env = Monitor(gym.make(ENV_ID))
    eval_env.reset(seed=10_000 + seed)
    cb = EvalCallback(eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES,
                      deterministic=True, log_path=out, best_model_save_path=None, verbose=0)

    model = DQN("MlpPolicy", env, seed=seed, verbose=0, **hp)
    t0 = time.time()
    model.learn(total_timesteps=N_TIMESTEPS, callback=cb)
    train_time = time.time() - t0
    model.save(os.path.join(out, "model"))

    # Фінальне оцінювання: детермінована (жадібна) політика, фіксовані seed епізодів
    test_env = gym.make(ENV_ID)
    returns, lengths = [], []
    for i in range(N_FINAL_EPISODES):
        obs, _ = test_env.reset(seed=50_000 + i)
        done, R, L = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = test_env.step(int(action))
            R += r; L += 1; done = term or trunc
        returns.append(R); lengths.append(L)

    n_params = sum(p.numel() for p in model.q_net.parameters())
    with open(os.path.join(out, "final.json"), "w") as f:
        json.dump(dict(config=name, seed=seed, hyperparams={k: (v if not isinstance(v, dict) else str(v)) for k, v in hp.items()},
                       train_time_s=train_time, final_returns=returns, final_lengths=lengths,
                       q_net_params=int(n_params)), f)
    return name, seed, f"{train_time:.0f}s, mean {np.mean(returns):.1f}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    jobs = [(name, seed, a.out) for seed in SEEDS for name in CONFIGS]
    with Pool(a.workers) as pool:
        for name, seed, msg in pool.imap_unordered(run_one, jobs):
            print(f"{name:14s} seed {seed}: {msg}", flush=True)
