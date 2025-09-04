import os
import json
from copy import deepcopy
import numpy as np
import transforms3d as t3d

from mimiclabs.mimiclabs.utils import get_robosuite_version

import robosuite
import robomimic.envs.env_base as EB


def get_controller_cfg(controller_types, controller_overrides, robots):
    """
    Get the controller config for the given controller type.
    """
    if get_robosuite_version() < "1.5":

        # update controller config with overrides
        controller_configs = []
        for i_robot in range(len(controller_overrides)):
            # load base controller config
            cfg_path = os.path.join(
                robosuite.__path__[0],
                "controllers",
                "config",
                f"{controller_types[i_robot]}.json",
            )
            with open(cfg_path) as f:
                controller_cfg = json.load(f)
            # update controller config
            controller_cfg.update(controller_overrides[i_robot])
            controller_configs.append(controller_cfg)
        return controller_configs
    else:
        assert isinstance(controller_overrides, list)
        # 0 is "right" arm, 1 is "left" arm
        robot_idx_to_arm_name = {0: "right", 1: "left"}

        # load base composite controller config
        base_composite_cfg_path = os.path.join(
            robosuite.__path__[0],
            "controllers",
            "config",
            "default",
            "composite",
            "basic.json",
        )
        with open(base_composite_cfg_path) as f:
            composite_controller_cfg = json.load(f)
        # remove all body parts
        composite_controller_cfg["body_parts"] = {}

        for i_robot in range(len(controller_overrides)):
            # load base controller config
            cfg_path = os.path.join(
                robosuite.__path__[0],
                "controllers",
                "config",
                "robots",
                f"default_{robots[i_robot].lower()}.json",
            )
            with open(cfg_path) as f:
                controller_cfg = json.load(f)["body_parts"]["arms"]["right"]
            # update controller config with overrides
            controller_cfg.update(controller_overrides[i_robot])
            # update composite controller config
            arm_name = robot_idx_to_arm_name[i_robot]
            composite_controller_cfg["body_parts"][arm_name] = controller_cfg

        return composite_controller_cfg


class RobosuiteTeleop:
    def __init__(
        self,
        env_name,
        robots=["Panda"],
        controller_types=["osc_pose"],
        controller_overrides=[
            {"control_delta": False}
        ],  # list of dicts # robosuite v1.4
        has_renderer=True,  # enable/disable rendering while stepping
        **kwargs,
    ):
        # Setup controller configs for each robot
        controller_configs = get_controller_cfg(
            controller_types, controller_overrides, robots
        )

        if get_robosuite_version() < "1.5":
            self.control_delta = controller_overrides[0]["control_delta"]
        else:
            # robosuite v1.5 uses "input_type": "delta" to specify delta actions
            self.control_delta = controller_overrides[0]["input_type"] == "delta"

        self._env_name = env_name
        self._env_kwargs = dict(
            robots=robots,
            controller_configs=(
                controller_configs[0]
                if len(controller_configs) == 1
                else controller_configs
            ),
            has_renderer=has_renderer,
            **kwargs,
        )
        self.env = robosuite.make(self._env_name, **self._env_kwargs)
        self._robots = self.env.robots

        for robot in self._robots:
            if get_robosuite_version() < "1.5":
                # forcing robosuite robot, using SingleArm._visualize_grippers() API in self.render(mode="human")
                assert isinstance(robot, robosuite.robots.single_arm.SingleArm)

    def serialize(self):
        env_kwargs = deepcopy(self._env_kwargs)

        # set control_delta=True for all robots since "actions" are delta actions
        controller_configs = env_kwargs["controller_configs"]
        if isinstance(controller_configs, (list, tuple)):
            for i in range(len(controller_configs)):
                controller_configs[i]["control_delta"] = True
        else:
            controller_configs["control_delta"] = True

        # set has_renderer to False
        env_kwargs["has_renderer"] = False

        return dict(
            env_name=self._env_name,
            type=EB.EnvType.ROBOSUITE_TYPE,  # needed for MimicGen/Robomimic, not the underlying env
            env_kwargs=env_kwargs,
        )

    def get_state(self):
        """
        Get current environment simulator state
        """
        state = np.array(self.env.sim.get_state().flatten())
        xml = self.env.sim.model.get_xml()
        return state, xml

    def _get_robosuite_arm_pose(self, robot):
        # get pos and ori of "grip_site"
        if get_robosuite_version() < "1.5":
            grip_site_id = robot.sim.model.site_name2id(robot.controller.eef_name)
        else:
            grip_site_id = robot.eef_site_id["right"]
        ee_pos = np.array(robot.sim.data.site_xpos[grip_site_id])
        ee_ori_mat = np.array(robot.sim.data.site_xmat[grip_site_id].reshape([3, 3]))
        pos_ori_mat = np.eye(4)
        pos_ori_mat[:3, :3] = ee_ori_mat
        pos_ori_mat[:3, -1] = ee_pos
        return pos_ori_mat

    def get_observation(self, di=None):
        if di is None:
            di = self.env._get_observations()

            # maybe flip images vertically
            flip_img = 1
            if robosuite.macros.IMAGE_CONVENTION == "opengl":
                flip_img = -1  # flip image to convert to OpenCV convention
            for k in di.keys():
                if k.endswith("_image"):  # robosuite image keys end with "_image"
                    di[k] = di[k][::flip_img, :, :]

        ret = deepcopy(di)

        return ret

    @property
    def last_eef_pose(self):
        """
        Returns list of pos_ori_mat's containing eef pose of each robot.
        """
        eef_poses = []
        for robot in self._robots:
            pos_ori_mat = self._get_robosuite_arm_pose(robot)
            eef_poses.append(pos_ori_mat)
        return eef_poses

    def step(self, controller_states):
        """
        controller_states: list of dicts, one for each robot
        """
        action = []
        action_abs = []
        for i_robot in range(len(self._robots)):
            #  Computing absolute action
            action_abs_pos = controller_states[i_robot]["target_pose"][:3]
            _vec, _theta = t3d.quaternions.quat2axangle(
                controller_states[i_robot]["target_pose"][3:]
            )
            action_axangle = _theta * _vec
            action_abs += (
                action_abs_pos
                + list(action_axangle)
                + controller_states[i_robot]["gripper_act"]
            )

            # Computing delta action
            action_pos = list(controller_states[i_robot]["delta_pos"])
            _vec, _theta = t3d.quaternions.quat2axangle(
                controller_states[i_robot]["delta_ori"]
            )
            action_axangle = _theta * _vec
            action += (
                action_pos
                + list(action_axangle)
                + controller_states[i_robot]["gripper_act"]
            )

        if self.control_delta:
            self.env.step(action)
        else:
            self.env.step(action_abs)

        return action, action_abs

    def reset(self):
        di = self.env.reset()
        return self.get_observation(di)

    def render(self, mode="human", height=None, width=None, camera_name=None):
        if mode == "human":
            # toggle gripper visualization
            for robot in self._robots:
                robot._visualize_grippers(True)

            if camera_name is not None:
                cam_id = self.env.sim.model.camera_name2id(camera_name)
                self.env.viewer.set_camera(cam_id)  # set OpenCVRenderer camera
            # NOTE by default this uses self.env.render_camera to render the scene,
            # which is not the same as the camera used for saving demos
            self.env.render()

            # toggle gripper visualization
            for robot in self._robots:
                robot._visualize_grippers(False)

        elif mode == "rgb_array":
            flip_img = 1
            if robosuite.macros.IMAGE_CONVENTION == "opengl":
                flip_img = -1  # flip image to convert to opencv convention

            if camera_name is None:
                camera_name = self.env.camera_names[0]
            return self.env.sim.render(
                height=height, width=width, camera_name=camera_name
            )[::flip_img]
        else:
            raise NotImplementedError(f"mode={mode} is not implemented")

    def done(self):
        return self.env._check_success()
