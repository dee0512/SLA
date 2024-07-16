import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from layers import get_layers
import TD3
from hyperparameters import get_hyperparameters
from common import make_env, make_env_cc

env_name = "LunarLanderContinuous-v2"
layers = get_layers(env_name)
num_layers = layers['layers']
hy = get_hyperparameters(env_name)
timestep = hy['timestep']
env = make_env_cc(env_name, 0, clock_wrapper=layers['clock'], clock_dim=layers['clock_dim'],
               prev_action_wrapper=layers['previous_action'])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
p = 0.5
kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": hy['discount'],
          "tau": hy['tau'],
          "policy_noise": hy['policy_noise'] * max_action, "noise_clip": hy['noise_clip'] * max_action,
          "policy_freq": hy['policy_freq'], "neurons": [400, 300]}

dfall = pd.DataFrame(columns=["seed", "noise", "avg. Reward"])

for seed in tqdm(range(10)):
    policies = []
    augment_type = "sla"
    arguments = [augment_type, env_name, seed, p]
    file_name = '_'.join([str(x) for x in arguments])
    for l in range(num_layers):
        k = copy.deepcopy(kwargs)
        if l < num_layers - 1:
            k["removed_indices"] = layers['layer' + str(l + 1)]
            policy = TD3.TempoRLTLA(**k)
        else:
            policy = TD3.TD3(**k)
        policy.load(f"./models/{file_name}_l{l+1}_layer_best")
        policies.append(policy)

    for i in range(21):
        env = make_env_cc(env_name, seed + 100, clock_wrapper=layers['clock'], clock_dim=layers['clock_dim'],
                       prev_action_wrapper=layers['previous_action'])
        noise = i * 0.05
        eval_reward = 0
        for i in range(10):
            eval_state, eval_done = env.reset(), False
            while not eval_done:

                removed_dim = -1
                if np.random.random() < noise:
                    # removed_dim = np.random.randint(2)
                    # if removed_dim == 0:
                    removed_dim = 0
                    # eval_state[0] = 0
                    # else:
                    eval_state[0:2] = 0

                if removed_dim == -1:
                    eval_action = policies[1].select_action(eval_state)
                else:
                    eval_action = policies[0].select_action(eval_state)
                eval_next_state, eval_r, eval_done, _ = env.step(eval_action)
                eval_reward += eval_r
                eval_state = eval_next_state

        avg_reward = eval_reward / 10
        dfall.loc[len(dfall)] = [seed, noise, avg_reward]

dfsla = dfall
dfsla.to_csv('llvelpos_dfsla.csv')
