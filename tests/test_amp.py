import pytest
import time
import numpy as np

from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3 import SAC

N_STEPS_TRAINING = 5000
SEED = 0

@pytest.parametrize("algo", SAC)
def test_deterministic_training_common(algo):
    results = [[], []]
    rewards = [[], []]
    # Smaller network
    kwargs_amp = {"policy_kwargs": dict(net_arch=[64]), "use_amp" : True}
    kwargs_reg = {"policy_kwargs": dict(net_arch=[64])}
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
    mixed_precision_time = time.time()
    model_amp.learn(N_STEPS_TRAINING)
    mixed_precision_time = time.time() - mixed_precision_time
    regular_time = time.time()
    model_reg.learn(N_STEPS_TRAINING)
    regular_time = time.time() - regular_time
    print("------------------------------------------------",
          "Mixed Precision Time: {:.2f} seconds".format(mixed_precision_time),
          "Regular Precision Time: {:.2f} seconds".format(regular_time),
    "------------------------------------------------")

if __name__ == "__main__":
    test_deterministic_training_common(SAC) #For dev use as of 1/20/24 TODO: Remove this when done