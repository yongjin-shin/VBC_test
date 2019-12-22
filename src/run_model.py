import argparse
import numpy as np
from smac.env import StarCraft2Env

# import whatever you need to defined and load your trained model
##########################################################################
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from main import recursive_dict_update, _get_config
from components.transforms import OneHot
from types import SimpleNamespace as SN
from copy import deepcopy
from utils.logging import get_logger
from components.episode_buffer import ReplayBuffer
import sys
import os
import yaml
import torch as th
##########################################################################


def test_model(model, env, num_runs=50):
    """
    :param model:
    :param env:
    :param num_runs:
    :return:
    """
    wins = []
    for i in range(num_runs):

        ############### you can design your test routines on here. ###############
        # A routine of yours to infer the action(s) from the state or observation
        # actions = model(state)

        print(wins)
        terminated = False

        learner = model
        runner = learner.runner_for_test

        runner.batch = runner.new_batch()
        env.reset()
        runner.t = 0

        episode_return = 0
        learner.mac.init_hidden(batch_size=runner.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [env.get_state()],
                "avail_actions": [env.get_avail_actions()],
                "obs": [env.get_obs()]
            }

            runner.batch.update(pre_transition_data, ts=runner.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, _, _, _ = learner.mac.select_actions_comm_proto(
                runner.batch, t_ep=runner.t, t_env=runner.t, test_mode=True)

            # ====Moved this Code from outside the loop into the loop=================
            reward, terminated, env_info = env.step(actions[0])
            # ========================================================================

            episode_return += reward

            post_transition_data = {
                "actions": actions[0],
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            runner.batch.update(post_transition_data, ts=runner.t)

            runner.t += 1

        last_data = {
            "state": [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs": [env.get_obs()]
        }
        runner.batch.update(last_data, ts=runner.t)

        # Select actions in the last stored state
        actions, _, _, _ = learner.mac.select_actions_comm_proto(
                runner.batch, t_ep=runner.t, t_env=runner.t, test_mode=True)
        runner.batch.update({"actions": actions}, ts=runner.t)
        info = env_info
        if info['battle_won'] is False:
            print("HERE")
        ##########################################################################

        # ===The following line is moved into "While" loop=========================
        # reward, terminated, info = env.step(actions)
        # =========================================================================

        if terminated:
            win = True if info['battle_won'] else False
        wins.append(float(win))

        env.save_replay()
        # import time
        # time.sleep(0.5)

    return np.average(wins)


def load_model(model_path):
    """
    :param model_path:
    :return: the loaded model (an instance of torch.nn.Module expected)
    """

    """
        do whatever you want to load model. 
        In the end of this code block, we expect to get loaded_model = model
    """
    ############### you can design your test routines on here. ###############
    params = ['--config=qmix_8m', '--env-config=sc2']
    logger = get_logger()

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    if config_dict["use_cuda"] and not th.cuda.is_available():
        config_dict["use_cuda"] = False
    config_dict['env_args']['map_name'] = config_dict['mac'][10:]
    config_dict['runner'] = 'episode'
    config_dict['batch_size_run'] = 1
    args = SN(**config_dict)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    learner.get_runner(runner)
    
    if args.use_cuda:
        learner.cuda()

    learner.load_models(model_path)
    ##########################################################################

    loaded_model = learner
    return loaded_model


def main(model_path):
    """
    Todo
    1) modify the difficulty setting
    2) window_size settings
    """
    env = StarCraft2Env(map_name="8m",
                        difficulty="9",
                        window_size_x=1920/3,
                        window_size_y=1200/3)
    loaded_model = load_model(model_path)
    mean_wr = test_model(loaded_model, env, num_runs=50)
    return mean_wr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='path to your model')
    args = parser.parse_args()
    mean_wr = main(args.path)

    print("Your average winning rate is {}".format(mean_wr))
