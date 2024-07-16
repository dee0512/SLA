import copy

import numpy as np
import torch
import argparse
import neptune.new as neptune
from layers import get_layers
import TD3
import utils
from hyperparameters import get_hyperparameters
from common import make_env, create_folders, make_env_cc


# Main function of the policy. Model is trained and evaluated inside.
def train(env_name="MountainCarContinuous-v0", seed=0):

    hy = get_hyperparameters(env_name)
    layers = get_layers(env_name)
    run = neptune.init(
        project="dee0512/Reflex",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzE3ZTdmOS05MzJlLTQyYTAtODIwNC0zNjAyMzIwODEzYWQifQ==",
    )
    env_type = hy['type']
    clock_dim = layers["clock_dim"]
    augment_type = "TD3"
    arguments = [augment_type, env_name, seed]
    file_name = '_'.join([str(x) for x in arguments])

    parameters = {
        'type': augment_type,
        'env_name': env_name,
        'seed': seed,
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    if env_type == 'mujoco':
        env = make_env(env_name, seed, clock_wrapper=layers['clock'], clock_dim=clock_dim, prev_action_wrapper=layers['previous_action'])
    else:
        timestep = default_timestep
        env = make_env_cc(env_name, seed, timestep, clock_wrapper=layers['clock'], clock_dim=clock_dim,
                          prev_action_wrapper=layers['previous_action'])

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    max_timesteps = hy['max_timesteps']
    eval_freq = hy['eval_freq']
    start_timesteps = hy['start_timesteps']

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": hy['discount'],
              "tau": hy['tau'],
              "policy_noise": hy['policy_noise'] * max_action, "noise_clip": hy['noise_clip'] * max_action,
              "policy_freq": hy['policy_freq'], "neurons": [400, 300]}

    policy = TD3.TempoRLTLA(**kwargs)
    replay_buffer = utils.ReplayBuffer(kwargs["state_dim"], action_dim, max_size=hy['replay_size'])

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_timestep = 1000

    best_performance = -10000

    evaluations = []

    # gives error if skip2 is not set for the first timestep
    if env == 'MountainCarContinuous-v2':
        noise = utils.OrnsteinUhlenbeckActionNoise(np.zeros(action_dim), np.ones(action_dim) * hy['expl_noise'])
    else:
        noise = lambda: np.random.normal(0, max_action * hy['expl_noise'], size=action_dim)

    for t in range(int(max_timesteps)):
        # Select action randomly or according to policy

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(state) + noise()).clip(-max_action, max_action)



        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_timesteps += 1
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_num += 1
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            if env_type == 'mujoco':
                eval_env = make_env(env_name, seed + 100, clock_wrapper=layers['clock'], clock_dim=clock_dim, prev_action_wrapper=layers['previous_action'])
            else:
                eval_env = make_env_cc(env_name,  seed + 100, timestep, clock_wrapper=layers['clock'], clock_dim=clock_dim, prev_action_wrapper=layers['previous_action'])
            task_reward = 0
            for _ in range(10):
                eval_state, eval_done = eval_env.reset(), False
                eval_episode_timesteps = 0
                while not eval_done:
                    eval_action = policy.select_action(eval_state)
                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_state = eval_next_state
                    eval_episode_timesteps += 1
                    task_reward += eval_reward
            avg_reward = task_reward / 10
            evaluations.append(avg_reward)
            print('------------------------------------------')
            print(f"Eval Reward: {avg_reward:.3f}")
            print(
                f" ------------------------------------------")
            run['avg_reward'].log(avg_reward)

            np.save(f"./results/{file_name}", evaluations)

            if best_performance <= avg_reward:
                best_performance = avg_reward
                run['best_reward'].log(best_performance)
                policy.save(f"./models/{file_name}_best")

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, hy['batch_size'])

    policy.save(f"./models/{file_name}_final")


    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="MountainCarContinuous-v0", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
