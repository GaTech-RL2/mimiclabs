"""
Script for collecting teleop demonstrations in a simulation environment 
using a Meta Quest controller or a SpaceMouse.

Meta Quest Controls:
    - Right controller: move robot
    - Left controller: move second robot (if applicable)
    - A button: save demo at any stage (saving also automatically happens when task is done)
    - B button: delete current demo, and start a new one
    - ctrl+C: discard current demo and exit
SpaceMouse Controls:
    - Joystick: move robot
    - Right button: start data collection / discard demo
    - Left button: toggle gripper
    - ctrl+C: discard current demo and exit

Args:
    --task_suite_name (optional, default=None): name of a MimicLabs task suite to generate demos for
    --task_name: name of task to generate demos for (if task_suite_name is None, then this should be a robosuite task name)
    --control_delta (optional, default=False): whether to control robot using delta OSC for data collection
    --robots (optional): robot(s) to use for this task
    --device (optional, default=quest): device to use for data collection
    --collect_more (optional, default=5): number of extra timesteps to collect after task success

Example usage:

    For a MimicLabs task:

    python scripts/collect_data.py \
        --task_suite_name example_suite \
        --task_name example_task \
        --robots Panda \
        --control_delta \
        --device quest

    python scripts/collect_data.py \
        --task_suite_name example_suite \
        --task_name example_task \
        --robots Panda \
        --control_delta \
        --device spacemouse

    For Robosuite tasks:

        python scripts/collect_data.py \
            --task_name Lift \
            --robots Panda \
            --control_delta \
            --device quest

        python scripts/collect_data.py \
            --task_name TwoArmTransport \
            --robots Panda Panda \
            --control_delta \
            --device quest
"""

import os
import time
import json
import argparse
from termcolor import colored

import mimiclabs
from mimiclabs.mimiclabs.utils import get_robosuite_version
import mimiclabs.mimiclabs.envs.bddl_utils as BDDLUtils
from mimiclabs.mimiclabs.envs.bddl_base_domain import BDDLBaseDomain, TASK_MAPPING
from mimiclabs.data_collection.sim.robosuite_teleop import RobosuiteTeleop
from mimiclabs.data_collection.sim.demo_saver import DemoSaver


def main(args):
    # Setup env kwargs for RobosuiteTeleop.
    kwargs = {}
    if args.task_suite_name is not None:  # task_name belongs to a MimicLabs task suite
        bddl_file_name = os.path.join(
            mimiclabs.__path__[0],
            "mimiclabs",
            "task_suites",
            args.task_suite_name,
            f"{args.task_name}.bddl",
        )
        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_name)
        env_name = TASK_MAPPING[parsed_problem["problem_name"]].__name__
        kwargs["bddl_file_name"] = bddl_file_name
    else:  # task_name is a robosuite env name
        env_name = args.task_name
    # Robots for the task.
    if args.robots is not None:
        robots = args.robots
    else:
        robots = ["Panda"]
    # Controller settings.
    controller_overrides = json.load(
        open(
            os.path.join(
                mimiclabs.__path__[0],
                "data_collection",
                "sim",
                "devices",
                "controller_configs",
                f"{args.device}.json",
            )
        )
    )
    if get_robosuite_version() < "1.5":
        # robosuite v1.4 uses "control_delta" to specify delta actions
        controller_overrides["control_delta"] = args.control_delta
    else:
        # robosuite v1.5 uses "input_type" in ["delta", "absolute"]
        controller_overrides["input_type"] = (
            "delta" if args.control_delta else "absolute"
        )
    # Put together kwargs for teleop env.
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_types=["osc_pose"] * len(robots),
        controller_overrides=[controller_overrides] * len(robots),
        horizon=args.horizon,
        **kwargs,
    )

    # Create teleop env.
    teleop_env = RobosuiteTeleop(**env_kwargs)
    env_args = teleop_env.serialize()

    # Create controller agent.
    if args.device == "quest":
        from mimiclabs.data_collection.sim.devices.quest_agent import QuestAgent

        teleop_agent = QuestAgent(robot_interface=teleop_env)
    elif args.device == "spacemouse":
        from mimiclabs.data_collection.sim.devices.spacemouse_agent import (
            SpaceMouseAgent,
        )

        teleop_agent = SpaceMouseAgent(robot_interface=teleop_env)

    # Demo saving.
    if args.task_suite_name is None:
        save_dir = os.path.join(os.path.dirname(__file__), "../demos", args.task_name)
    else:
        save_dir = os.path.join(
            os.path.dirname(__file__), "../demos", args.task_suite_name, args.task_name
        )
    os.makedirs(save_dir, exist_ok=True)
    demo_idx = 0

    while True:
        print("resetting env...")
        obs = teleop_env.reset()
        state, init_xml = teleop_env.get_state()
        teleop_env.render()

        # Demo saving.
        save_path = os.path.join(save_dir, f"demo_{demo_idx}.hdf5")
        while os.path.exists(save_path):
            demo_idx += 1
            save_path = os.path.join(save_dir, f"demo_{demo_idx}.hdf5")
        demo_saver = DemoSaver(save_path, env_args, init_xml, flush_freq=50)
        data_buffer = {
            "obs": None,
            "actions": None,
            "actions_abs": None,
            "states": None,
        }

        teleop_agent.reset_internal_state()
        delete_demo = False
        save_demo = False

        curr_demo_state_idx = 0
        demo_state_printed = False

        collect_n_more = None

        try:
            while True:

                if save_demo or delete_demo:
                    break

                if collect_n_more is not None:
                    if collect_n_more > 0:
                        collect_n_more -= 1
                    else:
                        save_demo = True
                        continue

                if teleop_env.done() and collect_n_more is None:
                    collect_n_more = args.collect_more
                    print(
                        colored(
                            f"Task done. Collecting {collect_n_more} more timesteps...",
                            "yellow",
                        )
                    )

                if isinstance(teleop_env.env, BDDLBaseDomain):
                    demo_states = teleop_env.env.parsed_problem["demonstration_states"]
                    if curr_demo_state_idx < len(demo_states):  # otherwise, done
                        curr_demo_state = demo_states[curr_demo_state_idx]
                        done = teleop_env.env._eval_predicate(curr_demo_state)
                        if not done and not demo_state_printed:
                            print(
                                colored("Current subtask:", "green"),
                                colored(
                                    f"{' '.join(curr_demo_state)}",
                                    "red",
                                    attrs=["bold"],
                                ),
                            )
                            demo_state_printed = True
                        elif done:
                            print(colored(f"Done.", "green", attrs=["bold"]))
                            curr_demo_state_idx += 1
                            demo_state_printed = False

                controller_state = teleop_agent.get_controller_state()
                if controller_state:  # sleep if None
                    save_demo = controller_state.get("save_demo", False)
                    delete_demo = controller_state.get("delete_demo", False)

                    if len(teleop_env._robots) == 1:  # single-arm
                        # data collection and env stepping only when controller is engaged
                        if controller_state["r"]["engaged"]:
                            ac, ac_abs = teleop_env.step([controller_state["r"]])
                            # demo saving
                            data_buffer["obs"] = obs
                            data_buffer["actions"] = ac
                            data_buffer["actions_abs"] = ac_abs
                            data_buffer["states"] = state
                            demo_saver.append(data_buffer)
                            # render to screen
                            teleop_env.render()
                    else:  # bimanual
                        # data collection starts automatically
                        ac, ac_abs = teleop_env.step(
                            [controller_state["r"], controller_state["l"]]
                        )
                        # demo saving
                        data_buffer["obs"] = obs
                        data_buffer["actions"] = ac
                        data_buffer["actions_abs"] = ac_abs
                        data_buffer["states"] = state
                        demo_saver.append(data_buffer)
                        # render to screen
                        teleop_env.render()

                time.sleep(0.001)

                obs = teleop_env.get_observation()
                state, _ = teleop_env.get_state()

            if delete_demo:
                print(colored("Discarding current demo...", "red"))
                demo_saver.discard()
            else:
                demo_saver.done()
                demo_idx += 1

        except KeyboardInterrupt:
            print(colored("Discarding current demo...", "red"))
            demo_saver.discard()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default=None,
        help="name of a MimicLabs task suite to generate demos for",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="name of task to generate demos for (if task_suite_name is None, then this should be a robosuite task name)",
    )
    parser.add_argument(
        "--control_delta",
        action="store_true",
        help="whether to control robot using delta OSC for data collection",
    )
    parser.add_argument(
        "--robots", nargs="+", default=None, help="robot(s) to use for this task"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="horizon for the environment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="quest",
        help="device to use for data collection; choices are 'quest' and 'spacemouse'",
    )
    parser.add_argument(
        "--collect_more",
        type=int,
        default=5,
        help="number of extra timesteps to collect after task success",
    )
    args = parser.parse_args()
    main(args)
