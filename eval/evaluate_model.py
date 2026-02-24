
import os
from typing import cast

import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.env
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent

def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    per_step_loss_threshold: np.ndarray,
    max_loss_threshold: np.ndarray,
):
    with torch.no_grad():
        num_added = 0
        sampling_round = 0
        transitions = replay_buffer.sample(batch_size=batch_size).astuple()
        obs_dataset = transitions[0]
        quantization_bits = np.log2(1e-6)

        total_initial_states = 0
        complete_rollouts = []
        while num_added < batch_size:
            batch_size_ = batch_size - int(num_added)
            total_initial_states += batch_size_

            if sampling_round > 0:
                batch = replay_buffer.sample(batch_size_)
                initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
                obs = initial_obs
            else:
                obs = obs_dataset
            obs_dims = obs.shape[-1]

            lost_info = np.zeros((batch_size_, obs_dims + 1))

            model_state = {}
            rollout_tracker = np.zeros(batch_size_)
            complete_rollouts.append(rollout_tracker)
            curr_idx = np.arange(batch_size_)
            for step in range(rollout_horizon):
                accum_dones = np.zeros(obs.shape[0], dtype=bool)
                action = agent.act(obs, sample=sac_samples_action, batched=True)
                model_state["obs"] = obs
                pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
                    action, model_state, sample=True
                )
                truncateds = np.zeros_like(pred_dones, dtype=bool)
                conditional_var = next_model_state["conditional_var"].cpu().numpy()

                lost_info_this_step = np.clip(
                    0.5 * np.log2(2 * np.pi * np.exp(1) * conditional_var) - quantization_bits, 0, np.inf
                )
                lost_info += lost_info_this_step

                accum_dones |= (lost_info_this_step > per_step_loss_threshold).any(axis=-1)
                accum_dones |= (lost_info > max_loss_threshold).any(axis=-1)

                sac_buffer.add_batch(
                    obs[~accum_dones],
                    action[~accum_dones],
                    pred_next_obs[~accum_dones],
                    pred_rewards[~accum_dones, 0],
                    pred_dones[~accum_dones, 0],
                    truncateds[~accum_dones, 0],
                    lost_info_this_step[~accum_dones],
                    lost_info[~accum_dones],
                    np.ones_like(lost_info[~accum_dones]),
                )
                num_added += (~accum_dones).sum()

                rollout_tracker[curr_idx[~accum_dones]] += 1

                accum_dones |= pred_dones.squeeze()
                obs = pred_next_obs[~accum_dones]
                if len(obs) == 0:
                    break
                lost_info = lost_info[~accum_dones]
                curr_idx = curr_idx[~accum_dones]

            print(num_added, step)
            sampling_round += 1
            if sampling_round == 1:
                assert num_added > 0
        rollout_tracker = np.concatenate(complete_rollouts)
        sac_buffer.re_compute_sampling_idxs()
        return num_added


load_path = "/home/bf863194/DSME/infoprop/exp/infoprop_dyna/infoprop_dyna_test/gym___HalfCheetah-v4/2026.02.24/141556"
sac_buffer_capacity = 500000

# ------------------- Load config -------------------
cfg = mbrl.util.common.load_hydra_cfg(load_path)

# ------------------- Create environment -------------------
env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

# ------------------- Load dynamics model -------------------
dynamics_model = mbrl.util.common.create_one_dim_tr_model(
    cfg, obs_shape, act_shape, model_dir=load_path
)

# ------------------- Load agent -------------------
mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
agent = SACAgent(
    cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
)
agent.sac_agent.load_checkpoint(
    ckpt_path=os.path.join(load_path, "sac_final.pth"), evaluate=True
)

# ------------------- Load replay buffer -------------------
use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
dtype = np.double if use_double_dtype else np.float32
replay_buffer = mbrl.util.common.create_replay_buffer(
    cfg,
    obs_shape,
    act_shape,
    obs_type=dtype,
    action_type=dtype,
    reward_type=dtype,
    load_dir=load_path,
)

# ------------------- Load thresholds -------------------
thresholds = np.load(os.path.join(load_path, "thresholds.npz"))
per_step_loss_threshold = thresholds["per_step_loss_threshold"]
max_loss_threshold = thresholds["max_loss_threshold"]

print(f"Loaded dynamics model, agent, replay buffer ({replay_buffer.num_stored} transitions), "
      f"and thresholds from {load_path}")
print(f"per_step_loss_threshold shape: {per_step_loss_threshold.shape}")
print(f"max_loss_threshold shape:      {max_loss_threshold.shape}")

# ------------------- Create model environment -------------------
torch_generator = torch.Generator(device=cfg.device)
model_env = mbrl.models.ModelEnv(
    env, dynamics_model, term_fn, None, generator=torch_generator
)

# ------------------- Create SAC buffer -------------------
rollout_batch_size = (
    cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
)
trains_per_epoch = int(
    np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
)
# Use the final rollout_schedule value to get end-of-training rollout length
num_epochs = cfg.overrides.num_steps // cfg.overrides.epoch_length
rollout_length = int(
    mbrl.util.math.truncated_linear(
        *(cfg.overrides.rollout_schedule + [num_epochs])
    )
)

rng = np.random.default_rng(seed=cfg.seed)
sac_buffer = mbrl.util.InfoReplayBuffer(sac_buffer_capacity, obs_shape, act_shape, rng=rng)

print(f"Created model_env and sac_buffer (capacity={sac_buffer_capacity}, rollout_length={rollout_length})")

# ------------------- Fill SAC buffer with model rollouts -------------------
while sac_buffer.num_stored < sac_buffer_capacity:

    num_added = rollout_model_and_populate_sac_buffer(
        model_env,
        replay_buffer,
        agent,
        sac_buffer,
        sac_samples_action=cfg.algorithm.sac_samples_action,
        rollout_horizon=rollout_length,
        batch_size=2048,
        per_step_loss_threshold=per_step_loss_threshold,
        max_loss_threshold=max_loss_threshold,
    )
    print(f"SAC buffer filled: {num_added} transitions added ({sac_buffer.num_stored} stored)")

    # then save the sac_buffer somewhere
