import torch as th
import numpy as np
import pytest
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC

N_STEPS_TRAINING = 5000
SEED = 0

@pytest.mark.parametrize("algo", [SAC])
def test_deterministic_training_common(algo=SAC):
    print("\nAMP Testing on device: " + th.cuda.get_device_name(0))
    results = [[], []]
    rewards = [[], []]
    # Smaller network
    kwargs_amp = {"policy_kwargs": dict(net_arch=[512,512,512]), "use_amp" : True, "batch_size" : 8192}
    kwargs_reg = {"policy_kwargs": dict(net_arch=[512,512,512]), "batch_size" : 8192}
    env_id = "Pendulum-v1"
    if algo in [SAC]:
        kwargs_amp.update(
            {"action_noise": NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)), "learning_starts": 100, "train_freq": 4}
        )
        kwargs_reg.update(
            {"action_noise": NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)), "learning_starts": 100, "train_freq": 4}
        )
    model_amp = algo("MlpPolicy", env_id, seed=SEED, **kwargs_amp)
    model_reg = algo("MlpPolicy", env_id, seed=SEED, **kwargs_reg)
    mixed_precision_start = th.cuda.Event(enable_timing=True)
    mixed_precision_end = th.cuda.Event(enable_timing=True)
    regular_start = th.cuda.Event(enable_timing=True)
    regular_end = th.cuda.Event(enable_timing=True)
    mixed_precision_start.record()
    model_amp.learn(N_STEPS_TRAINING)
    th.cuda.synchronize()
    mixed_precision_end.record()
    regular_start.record()
    model_reg.learn(N_STEPS_TRAINING)
    th.cuda.synchronize()
    regular_end.record()
    mixed_precision_time = mixed_precision_start.elapsed_time(mixed_precision_end)/1000
    regular_time = regular_start.elapsed_time(regular_end)/1000
    print("------------------------------------------------\n",
          "Mixed Precision Time: {:.2f} seconds\n".format(mixed_precision_time),
          "Regular Precision Time: {:.2f} seconds\n".format(regular_time),
          "Total time saved: {:.2f} seconds\n".format(regular_time - mixed_precision_time),
    "------------------------------------------------")
