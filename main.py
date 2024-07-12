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
def train(env_name="MountainCarContinuous-v0", seed=0, p=1):

    hy = get_hyperparameters(env_name)
    layers = get_layers(env_name)
    run = neptune.init(
        project="dee0512/Reflex",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzE3ZTdmOS05MzJlLTQyYTAtODIwNC0zNjAyMzIwODEzYWQifQ==",
    )
    env_type = hy['type']
    clock_dim = layers["clock_dim"]
    num_layers = layers['layers']
    augment_type = "sla"
    arguments = [augment_type, env_name, seed, p]
    file_name = '_'.join([str(x) for x in arguments])

    parameters = {
        'type': augment_type,
        'env_name': env_name,
        'seed': seed,
        'p': p,
    }
    run["parameters"] = parameters
    print("---------------------------------------")
    print(f"Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    if env_type == 'mujoco':
        env = make_env(env_name, seed, clock_wrapper=layers['clock'], clock_dim=clock_dim, prev_action_wrapper=layers['previous_action'])
    else:
        env = make_env_cc(env_name, seed, clock_wrapper=layers['clock'], clock_dim=clock_dim,
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

    policies = []
    buffers = []
    skip_buffers = []
    layer_evaluations = []
    eval_actions = []
    for l in range(num_layers):
        k = copy.deepcopy(kwargs)
        if l < num_layers - 1:
            k["removed_indices"] = layers['layer' + str(l + 1)]
            policy = TD3.TempoRLTLA(**k)
        else:
            policy = TD3.TD3(**k)
        policies.append(policy)
        replay_buffer = utils.ReplayBuffer(k["state_dim"], action_dim, max_size=hy['replay_size'])
        buffers.append(replay_buffer)
        if l < num_layers - 1:
            sb = utils.FiGARReplayBuffer(k["state_dim"], action_dim, 1, max_size=hy['replay_size'])
            skip_buffers.append(sb)
        layer_evaluations.append([])
        eval_actions.append([])

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_timestep = env.env.env._max_episode_steps


    best_performance = -10000

    layer_best_performance = np.ones(num_layers) * -10000
    evaluations = []

    train_actions = np.zeros(num_layers)
    layer_actions = np.zeros([num_layers, action_dim])
    layer_skips = np.zeros(num_layers-1)
    # gives error if skip2 is not set for the first timestep
    if env == 'MountainCarContinuous-v2':
        noise = utils.OrnsteinUhlenbeckActionNoise(np.zeros(action_dim), np.ones(action_dim) * hy['expl_noise'])
    else:
        noise = lambda: np.random.normal(0, max_action * hy['expl_noise'], size=action_dim)

    for t in range(int(max_timesteps)):
        # Select action randomly or according to policy

        for i, pol in enumerate(policies):

            if t < start_timesteps:
                a = env.action_space.sample()
                if i < num_layers-1:
                    skip = np.random.randint(2)
            else:
                a = (pol.select_action(state) + noise()).clip(-max_action, max_action)
                if i < num_layers-1:
                    skip = pol.select_skip(state, a)
                    if np.random.random() < hy['expl_noise']:
                        skip = np.random.randint(2)  # + 1 sonce randint samples from [0, max_rep)
                    else:
                        skip = np.argmax(skip)  # + 1 since indices start at 0
            action = a
            layer_actions[i] = a
            if i < num_layers-1:
                layer_skips[i] = skip
            if i < num_layers-1 and skip == 0:
                train_actions[i] += 1
                for j in range(i+1, num_layers):
                    layer_actions[j] = a
                    if j < num_layers -1:
                        layer_skips[j] = 0
                break

            elif i == num_layers-1:
                train_actions[i] += 1

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_timesteps += 1
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0
        for i, buffer in enumerate(buffers):
            buffer.add(state, layer_actions[i], next_state, reward - (p if ((i < num_layers-1) and layer_skips[i] > 0) else 0), done_bool)
            if i < num_layers -1:
                skip_buffers[i].add(state, layer_actions[i], layer_skips[i], next_state, layer_actions[i], reward - (p if (layer_skips[i] > 0) else 0), done_bool)

        state = next_state

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Actions:{train_actions} Reward: {episode_reward:.3f}")
            # Reset environment
            train_actions = np.zeros(num_layers)
            state, done = env.reset(), False
            episode_reward = 0
            episode_num += 1
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            if env_type == 'mujoco':
                eval_env = make_env(env_name, seed + 100, clock_wrapper=layers['clock'], clock_dim=clock_dim, prev_action_wrapper=layers['previous_action'])
            else:
                eval_env = make_env_cc(env_name,  seed + 100, clock_wrapper=layers['clock'], clock_dim=clock_dim, prev_action_wrapper=layers['previous_action'])
            task_reward = 0
            eval_num_actions = np.zeros(num_layers)
            for _ in range(10):
                eval_state, eval_done = eval_env.reset(), False
                eval_episode_timesteps = 0
                while not eval_done:
                    for i, pol in enumerate(policies):
                        eval_action = pol.select_action(eval_state)
                        if i < num_layers - 1:
                            eval_skip = np.argmax(pol.select_skip(eval_state, eval_action))
                            if eval_skip == 0:
                                eval_num_actions[i] += 1
                                break
                        else:
                            eval_num_actions[i] += 1

                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_state = eval_next_state
                    eval_episode_timesteps += 1
                    task_reward += eval_reward
            avg_reward = task_reward / 10
            evaluations.append(avg_reward)
            print('------------------------------------------')
            for i, ea in enumerate(eval_actions):
                ea.append(eval_num_actions[i]/10)
                print(f"l{i+1}: {eval_num_actions[i]/10:.3f}")
                run['avg_layer'+str(i+1)+'_actions'].log(eval_num_actions[i]/10)
                np.save(f"./results/{file_name}_layer{i+1}_actions", eval_num_actions[i]/10)

            print(f"Eval Reward: {avg_reward:.3f}")
            print(
                f" ------------------------------------------")
            run['avg_reward'].log(avg_reward)

            np.save(f"./results/{file_name}", evaluations)

            if best_performance <= avg_reward:
                best_performance = avg_reward
                run['best_reward'].log(best_performance)
                for i, pol in enumerate(policies):
                    pol.save(f"./models/{file_name}_l{i+1}_best")

            for i in range(num_layers):
                task_reward = 0
                for _ in range(10):
                    eval_state, eval_done = eval_env.reset(), False
                    eval_episode_timesteps = 0
                    while not eval_done:
                        eval_action = policies[i].select_action(eval_state)
                        eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                        eval_state = eval_next_state
                        eval_episode_timesteps += 1
                        task_reward += eval_reward
                layer_evaluations[i].append(task_reward/10)
                np.save(f"./results/{file_name}_layer{i}", layer_evaluations[i])
                if layer_best_performance[i] <= task_reward/10:
                    layer_best_performance[i] = task_reward/10
                    policies[i].save(f"./models/{file_name}_l{i+1}_layer_best")

                print(
                    f" --------------- Layer: {i+1}, Evaluation reward {task_reward/10:.3f}")
                run['Layer'+str(i+1)+' reward'].log(task_reward/10)

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            for i, pol in enumerate(policies):
                pol.train(buffers[i], hy['batch_size'])

            for i, pol in enumerate(policies):
                if i < num_layers-1:
                    pol.train_skip(skip_buffers[i], hy['batch_size'])


    for i, pol in enumerate(policies):
        pol.save(f"./models/{file_name}_l{i+1}_final")
        # buffers[i].save(f"./models/{file_name}_l{i + 1}_buffer")

    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="MountainCarContinuous-v0", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--p", default=1.0, type=float, help="reward penalty for the slow network")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
