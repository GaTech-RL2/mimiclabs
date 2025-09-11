"""
Script to extract observations from a dataset of states and actions.

Example usage:
    python scripts/dataset_states_to_obs.py \
        --dataset /path/to/dataset.hdf5 \
        --output_dir /path/to/output_dir \
        --camera_names agentview \
        --camera_height 128 \
        --camera_width 128 \
        --n 10 \
        --render
"""

import os
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import json

import robosuite

import robomimic.utils.tensor_utils as TensorUtils

from mimiclabs.mimiclabs.envs import *


def extract_trajectory(env, initial_state, states, actions, actions_abs, render=False):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        render (bool): whether to render the environment
    """
    # assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    obs = env.reset_to(initial_state)
    if render:
        # render to screen
        env.render()

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
    )
    if actions_abs is not None:
        traj["actions_abs"] = np.array(actions_abs)

    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})

        if render:
            # render to screen
            env.render()

        # infer reward signal
        r = env.reward()

        # infer done signal
        done = env._check_success()

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def dataset_states_to_obs(args):
    """
    Main function to convert a dataset of states and actions to a dataset of observations.
    """
    # open source hdf5 file
    f_src = h5py.File(args.dataset, "r")
    demos = list(f_src["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    # create environment with updated env_meta
    env_meta = json.loads(f_src["data"].attrs["env_args"])
    env_meta["env_kwargs"]["camera_heights"] = args.camera_height
    env_meta["env_kwargs"]["camera_widths"] = args.camera_width
    env_meta["env_kwargs"]["camera_names"] = args.camera_names
    env_meta["env_kwargs"]["camera_depths"] = args.camera_depths
    env_meta_to_save = deepcopy(env_meta)
    if args.render:
        env_meta["env_kwargs"]["has_renderer"] = True
    env = robosuite.make(
        env_meta["env_name"],
        **env_meta["env_kwargs"],
    )

    # create output hdf5 file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_name = os.path.basename(args.dataset)[:-5] + "_im.hdf5"
    output_file = os.path.join(args.output_dir, output_file_name)
    f_dst = h5py.File(output_file, "w")
    data_grp = f_dst.create_group("data")

    # copy global attributes
    for k in f_src["data"].attrs:
        if k == "env_args":
            # save env meta to file
            data_grp.attrs[k] = json.dumps(env_meta_to_save)
        else:
            # copy other attributes
            data_grp.attrs[k] = f_src["data"].attrs[k]

    # saving trajectories
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f_src["data/{}/states".format(ep)][()]
        initial_state = dict(
            states=states[0], model=f_src[f"data/{ep}"].attrs["model_file"]
        )

        # get actions from data
        actions = f_src["data/{}/actions".format(ep)][()]
        actions_abs = f_src["data/{}/actions_abs".format(ep)][()]

        traj = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            actions_abs=actions_abs,
            render=args.render,
        )

        # store transitions
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        for k in traj["obs"]:
            if args.compress:
                ep_data_grp.create_dataset(
                    "obs/{}".format(k),
                    data=np.array(traj["obs"][k]),
                    compression="gzip",
                )
            else:
                ep_data_grp.create_dataset(
                    "obs/{}".format(k), data=np.array(traj["obs"][k])
                )
            if not args.exclude_next_obs:
                if args.compress:
                    ep_data_grp.create_dataset(
                        "next_obs/{}".format(k),
                        data=np.array(traj["next_obs"][k]),
                        compression="gzip",
                    )
                else:
                    ep_data_grp.create_dataset(
                        "next_obs/{}".format(k), data=np.array(traj["next_obs"][k])
                    )

        # copy episode metadata
        ep_data_grp.attrs["model_file"] = initial_state[
            "model"
        ]  # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
            0
        ]  # number of transitions in this episode

    print(f"Wrote {len(demos)} trajectories to {output_file}")

    # close hdf5 files
    f_src.close()
    f_dst.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="name of output folder to write to",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_depths",
        action="store_true",
        help="(optional) store depth images in dataset",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    parser.add_argument(
        "--exclude_next_obs",
        action="store_true",
        help="(optional) exclude next obs in dataset",
    )

    parser.add_argument(
        "--compress",
        action="store_true",
        help="(optional) compress observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="(optional) render observations during playback",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
