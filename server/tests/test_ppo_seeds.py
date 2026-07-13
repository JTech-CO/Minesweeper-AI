from trainer.algorithms.ppo import PPOBackend
from trainer.config import ExperimentConfig


def test_parallel_env_and_reset_seeds_do_not_overlap(tmp_path) -> None:
    config = ExperimentConfig(
        algorithm="ppo",
        difficulty="beginner",
        device="cpu",
        width=16,
        blocks=1,
        num_envs=3,
        rollout_steps=1,
        batch_size=3,
        total_updates=1,
        eval_every=99,
        checkpoint_every=99,
    )
    backend = PPOBackend(config, tmp_path)
    initial = [env._seed for env in backend.envs]
    backend._reset_env(0)
    assert initial == [0, 1, 2]
    assert backend.envs[0]._seed == 3

