"""
Base environment class for BDDL-based tasks in MimicLabs (adapted from LIBERO).
"""

import os
import numpy as np
import cv2
import pathlib
import uuid
import time
import datetime

import mujoco

import libero

import mimiclabs
import mimiclabs.mimiclabs.macros as macros
from mimiclabs.mimiclabs import assets_root as ASSETS_ROOT
import mimiclabs.mimiclabs.envs.bddl_utils as BDDLUtils
from ..utils import *
from .utils import *
from .arenas import *
from .object_states import *
from .objects import *
from .regions import *

import robosuite

if get_robosuite_version() < "1.5":
    from robosuite.environments.manipulation.single_arm_env import (
        SingleArmEnv as RobosuiteEnv,
    )
else:
    from robosuite.environments.manipulation.manipulation_env import (
        ManipulationEnv as RobosuiteEnv,
    )
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor

try:
    # try to import mimicgen environments
    import mimicgen

    MIMICGEN_PATH = mimicgen.__path__[0]
except ImportError:
    print("WARNING: could not import mimicgen envs")

try:
    # try to import robocasa environments
    with disable_module_import("robocasa"):
        from robocasa.models import assets_root as robocasa_assets_root

        ROBOCASA_PATH = os.path.dirname(os.path.dirname(robocasa_assets_root))
except ImportError:
    print("WARNING: could not import robocasa envs")


MIMICLABS_TMP_FOLDER = macros.MIMICLABS_TMP_FOLDER

TASK_MAPPING = {}


def register_problem(target_class):
    TASK_MAPPING[target_class.__name__.lower()] = target_class


class BDDLBaseDomain(RobosuiteEnv):
    """
    A base domain for parsing bddl files.
    """

    def __init__(
        self,
        bddl_file_name,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        object_property_initializers=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        table_full_size=(1.0, 1.0, 0.05),
        workspace_offset=(0.0, 0.0, 0.0),
        arena_type="table",
        scene_xml="scenes/mimiclabs_scenes/lab2/lab2.xml",
        scene_properties={},
        use_depth_obs=False,  # NOTE(VS) unused; maybe replicate Robomimic's EnvRobosuite behavior
        **kwargs,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.workspace_offset = workspace_offset
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.conditional_placement_initializer = None
        self.conditional_placement_on_objects_initializer = None

        # object property initializer

        if object_property_initializers is not None:
            self.object_property_initializers = object_property_initializers
        else:
            self.object_property_initializers = list()

        # Keep track of movable objects in the tasks
        self.objects_dict = {}
        # Kepp track of fixed objects in the tasks
        self.fixtures_dict = {}
        # Keep track of site objects in the tasks. site objects
        # (instances of SiteObject)
        self.object_sites_dict = {}
        # This is a dictionary that stores all the object states
        # interface for all the objects
        self.object_states_dict = {}

        # For those that require visual feature changes, update the state every time step to avoid missing state changes. We keep track of this type of objects to make predicate checking more efficient.
        self.tracking_object_states_change = []

        self.object_sites_dict = {}

        self.objects = []
        self.fixtures = []
        # self.custom_material_dict = {}

        # Try to replace BDDL file path to be relative to MimicLabs installation.
        bddl_file_name = BDDLUtils.resolve_bddl_file_name(bddl_file_name)

        self.bddl_file_name = bddl_file_name
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)

        # Override table_full_size if specified in bddl
        if "table_params" in self.parsed_problem:
            if "size" in self.parsed_problem["table_params"]:
                table_size = self.parsed_problem["table_params"]["size"]
                # Set table length and width (used for resizing geoms in TableArena)
                self.table_full_size = (
                    table_size[0],
                    table_size[1],
                    self.table_full_size[2],
                )
                # Set table height in table_offset and workspace_offset
                self.table_offset = (
                    self.table_offset[0],
                    self.table_offset[1],
                    table_size[2],
                )
                self.workspace_offset = self.table_offset

        self.obj_of_interest = self.parsed_problem["obj_of_interest"]

        self._assert_problem_name()

        self._arena_type = arena_type
        self._arena_xml = (
            scene_xml
            if os.path.isabs(scene_xml)
            else os.path.join(ASSETS_ROOT, scene_xml)
        )
        self._arena_properties = scene_properties

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            **kwargs,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def edit_model_xml(self, xml_str):
        """
        Update from superclass to postprocess MimicLabs paths too.

        This function edits the model xml with custom changes, including resolving relative paths,
        applying changes retroactively to existing demonstration files, and other custom scripts.
        Environment subclasses should modify this function to add environment-specific xml editing features.
        Args:
            xml_str (str): Mujoco sim demonstration XML file as string
        Returns:
            str: Edited xml file as string
        """

        # replace mesh and texture file paths
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue
            old_path_split = old_path.split("/")

            # maybe replace all paths to mimicgen_envs assets
            check_lst = [
                loc for loc, val in enumerate(old_path_split) if val == "mimicgen"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = MIMICGEN_PATH.split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)
                continue  # path may contain "robosuite", hence continue

            # maybe replace all paths to robocasa assets
            check_lst = [
                loc for loc, val in enumerate(old_path_split) if val == "robocasa"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = ROBOCASA_PATH.split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # maybe replace all paths to mimiclabs assets
            check_lst = [
                loc for loc, val in enumerate(old_path_split) if val == "mimiclabs"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                if "robosuite" in old_path_split[ind + 1 :]:
                    new_path_split = (
                        os.path.split(mimiclabs.__path__[0])[0].split("/")
                        + ["mimiclabs"]
                        + old_path_split[ind + 1 :]
                    )
                else:
                    new_path_split = (
                        os.path.split(mimiclabs.__path__[0])[0].split("/")
                        + ["mimiclabs", "mimiclabs"]
                        + old_path_split[ind + 1 :]
                    )
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)
                continue  # path may contain "robosuite", hence continue

            # maybe replace all paths to robosuite assets
            check_lst = [
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = (
                    os.path.split(robosuite.__file__)[0].split("/")
                    + old_path_split[ind + 1 :]
                )
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # maybe replace all paths to libero assets
            check_lst = [
                loc for loc, val in enumerate(old_path_split) if val == "libero"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = (
                    os.path.split(libero.__path__[0])[0].split("/")
                    + ["libero", "libero"]
                    + old_path_split[ind + 1 :]
                )
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the task succeeds.

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _assert_problem_name(self):
        """Implement this to make sure the loaded bddl file has the correct problem name specification."""
        assert (
            self.parsed_problem["problem_name"] == self.__class__.__name__.lower()
        ), "Problem name mismatched"

    def _load_fixtures_in_arena(self, mujoco_arena):
        """
        Load fixtures based on the bddl file description. Please override the method in the custom problem file.
        """
        raise NotImplementedError

    def _load_objects_in_arena(self, mujoco_arena, object_params):
        """
        Load movable objects based on the bddl file description
        """
        raise NotImplementedError

    def _load_sites_in_arena(self, mujoco_arena):
        """
        Load sites information from each object to keep track of them for predicate checking
        """
        raise NotImplementedError

    def _generate_object_state_wrapper(
        self, skip_object_names=["main_table", "floor", "countertop", "coffee_table"]
    ):
        object_states_dict = {}
        tracking_object_states_changes = []
        for object_name in self.objects_dict.keys():
            if object_name in skip_object_names:
                continue
            joints = None
            if hasattr(self.get_object(object_name), "object_state_joints"):
                joints = self.get_object(object_name).object_state_joints
            joints = None
            if hasattr(self.get_object(object_name), "object_state_joints"):
                joints = self.get_object(object_name).object_state_joints
            object_states_dict[object_name] = ObjectState(
                self, object_name, joints=joints
            )
            if (
                self.objects_dict[object_name].category_name
                in VISUAL_CHANGE_OBJECTS_DICT
            ):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.fixtures_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(
                self, object_name, is_fixture=True
            )
            if (
                self.fixtures_dict[object_name].category_name
                in VISUAL_CHANGE_OBJECTS_DICT
            ):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.object_sites_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = SiteObjectState(
                self,
                object_name,
                parent_name=self.object_sites_dict[object_name].parent_name,
            )
        self.object_states_dict = object_states_dict
        self.tracking_object_states_change = tracking_object_states_changes

    def _setup_camera(self, mujoco_arena, pos=None, quat=None, camera_attribs=None):
        # Assert that either both pos and quat are provided, or neither is provided
        assert (pos is None and quat is None) or (
            pos is not None and quat is not None
        ), "Both pos and quat must be provided together, or neither should be provided"

        if pos is None and quat is None:
            # Default camera pose
            pos = [1.0, 0, 1.6]
            quat = [0.5963678, 0.3799282, 0.3799282, 0.5963678]

        # Set agentview camera in arena
        mujoco_arena.set_camera(
            camera_name="agentview", pos=pos, quat=quat, camera_attribs=camera_attribs
        )

        # Add additional cameras for visualization purposes
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="paperview",
            pos=[2.1, 0.535, 2.075],
            quat=[0.513, 0.353, 0.443, 0.645],
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # Adjust base pose accordingly

        if self._arena_type == "table":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.workspace_offset,
                table_friction=(0.6, 0.005, 0.0001),
                xml=self._arena_xml,
                **self._arena_properties,
            )
        else:
            raise NotImplementedError

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self._load_fixtures_in_arena(mujoco_arena)

        self._load_objects_in_arena(mujoco_arena, self.parsed_problem["object_params"])

        self._load_sites_in_arena(mujoco_arena)

        # Change camera pose in arena
        if len(self.parsed_problem["camera"].get("ranges", [])) > 0:
            cam_pos_quat = self._sample_camera_pose(
                options=self.parsed_problem["camera"]
            )
            camera_attribs = None
            if self.parsed_problem["camera"]["intrinsics"] is not None:
                camera_attribs = self.parsed_problem["camera"]["intrinsics"]
                for key, value in camera_attribs.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        camera_attribs[key] = " ".join([str(v) for v in value])
                    else:
                        camera_attribs[key] = str(value)
            self._setup_camera(
                mujoco_arena,
                pos=cam_pos_quat["pos"],
                quat=cam_pos_quat["quat"],
                camera_attribs=camera_attribs,
            )
        else:
            self._setup_camera(mujoco_arena)

        self._generate_object_state_wrapper()

        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        # randomize textures if specified in bddl
        self._randomize_object_textures(mujoco_arena)

        self._randomize_lighting_dir(mujoco_arena)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _sample_camera_pose(self, options):
        ranges_r_theta_phi = options["ranges"]
        range_choice = np.random.choice(range(len(ranges_r_theta_phi)))
        range_r_theta_phi = ranges_r_theta_phi[range_choice]
        range_r = [range_r_theta_phi[0], range_r_theta_phi[3]]
        range_theta = [range_r_theta_phi[1], range_r_theta_phi[4]]
        range_phi = [range_r_theta_phi[2], range_r_theta_phi[5]]

        jitter_mode = options["jitter_mode"]

        if jitter_mode == "uniform":
            sample_r = (range_r[1] - range_r[0]) * np.random.random_sample() + range_r[
                0
            ]
            sample_theta = (
                range_theta[1] - range_theta[0]
            ) * np.random.random_sample() + range_theta[0]
            sample_phi = (
                range_phi[1] - range_phi[0]
            ) * np.random.random_sample() + range_phi[0]
        elif jitter_mode == "normal":
            sample_r = np.clip(
                np.random.normal(
                    (range_r[1] + range_r[0]) / 2.0, (range_r[1] - range_r[0]) / 6.0
                ),
                range_r[0],
                range_r[1],
            )
            sample_theta = np.clip(
                np.random.normal(
                    (range_theta[1] + range_theta[0]) / 2.0,
                    (range_theta[1] - range_theta[0]) / 6.0,
                ),
                range_theta[0],
                range_theta[1],
            )
            sample_phi = np.clip(
                np.random.normal(
                    (range_phi[1] + range_phi[0]) / 2.0,
                    (range_phi[1] - range_phi[0]) / 6.0,
                ),
                range_phi[0],
                range_phi[1],
            )
        else:
            raise NotImplementedError(
                f"Camera jitter mode '{jitter_mode}' is not supported. "
                f"Supported modes are: 'uniform', 'normal'."
            )

        # Converting spherical to pos and quat
        r_theta_phi = (sample_r, sample_theta, sample_phi)
        if options["unit"] == "degrees":
            r_theta_phi = (
                r_theta_phi[0],
                np.deg2rad(r_theta_phi[1]),
                np.deg2rad(r_theta_phi[2]),
            )
        else:
            assert (
                options["unit"] == "radians"
            ), "Unit must be either 'radians' or 'degrees'"
        pos, quat_wxyz = convert_spherical_to_pos_quat(r_theta_phi)

        # Adjusting position based on look_at option
        if isinstance(options["look_at"], str):
            if options["look_at"] == "table":
                # look at table center
                cam_fixation_pos = np.array(self.table_offset)
            else:
                raise NotImplementedError(
                    "Only 'table' or 3D point are supported for look_at option."
                )
        else:
            # look at specified 3D point
            assert (
                len(options["look_at"]) == 3
            ), "look_at position must be a list of 3 floats"
            assert all(
                isinstance(coord, (int, float)) for coord in options["look_at"]
            ), "All coordinates in look_at position must be numbers"
            cam_fixation_pos = np.array(options["look_at"])
        pos += cam_fixation_pos

        return {"pos": pos, "quat": quat_wxyz}

    def _randomize_lighting_dir(self, mujoco_arena):
        lighting_params = self.parsed_problem["lighting"]
        light = mujoco_arena.worldbody.find("./light")
        if light is not None:
            # Setting shadow
            light.attrib["castshadow"] = str(
                lighting_params.get("shadow", False)
            ).lower()  # default: no shadow

            # Setting lighting direction
            ranges_r_theta_phi = lighting_params.get(
                "source", [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            )  # default: top-down light source
            range_choice = np.random.choice(range(len(ranges_r_theta_phi)))
            range_r_theta_phi = ranges_r_theta_phi[range_choice]
            range_r = [range_r_theta_phi[0], range_r_theta_phi[3]]
            range_theta = [range_r_theta_phi[1], range_r_theta_phi[4]]
            range_phi = [range_r_theta_phi[2], range_r_theta_phi[5]]
            sample_r = (range_r[1] - range_r[0]) * np.random.random_sample() + range_r[
                0
            ]
            sample_theta = (
                range_theta[1] - range_theta[0]
            ) * np.random.random_sample() + range_theta[0]
            sample_phi = (
                range_phi[1] - range_phi[0]
            ) * np.random.random_sample() + range_phi[0]
            pos, _ = convert_spherical_to_pos_quat((sample_r, sample_theta, sample_phi))
            light.attrib["dir"] = f"{-pos[0]} {-pos[1]} {-pos[2]}"

    def _randomize_object_textures(self, mujoco_arena):
        """
        The following texture types are supported:
            file, wood, color, fractal, jitter

        There is a special case for table texture of type "color".
        In this case, we set the table texture to a solid color
        by modifying the rgba attribute of the table visual geom
        directly, instead of changing the texture file.
        """
        for obj_name, texture_params in self.parsed_problem["textures"].items():
            # Find the texture element in the xml
            tex = None
            if "table" in obj_name:
                if texture_params["texture_type"] == "color":
                    # Special case for "color" table texture: set rgba directly
                    hsv_ranges = texture_params["hsv"]
                    hue, sat, val = sample_hsv_from_hsv_ranges(hsv_ranges)
                    rgba = hsv_to_rgba(hue, sat, val, alpha=1.0)
                    mujoco_arena.set_table_rgba(rgba=rgba)
                    continue  # skip texture file modification
                tex = mujoco_arena.asset.find("./texture[@name='tex-table']")
            elif obj_name in self.objects_dict:
                tex = self.objects_dict[obj_name].asset.find("./texture")
            elif obj_name in self.fixtures_dict:
                tex = self.fixtures_dict[obj_name].asset.find("./texture")

            if tex is not None:  # cannot use texture on obj otherwise
                tex_file = tex.attrib["file"]
                img_bgr = cv2.imread(tex_file)  # BGR
                H, W = img_bgr.shape[:2]

                if texture_params["texture_type"] == "file":
                    if "files" not in texture_params:
                        # use default texture files

                        if "table" in obj_name:
                            texture_folder = os.path.join(
                                ASSETS_ROOT, "scenes/mimiclabs_scenes/textures/tables"
                            )
                        else:
                            texture_folder = os.path.join(
                                ASSETS_ROOT, "scenes/mimiclabs_scenes/textures/object"
                            )
                        texture_files = os.listdir(texture_folder)
                        tex_file = np.random.choice(texture_files)
                        tex_file_path = os.path.join(texture_folder, tex_file)

                    else:
                        # use specified texture files
                        tex_file_path = os.path.join(
                            ASSETS_ROOT, np.random.choice(texture_params["files"])
                        )
                        assert os.path.exists(
                            tex_file_path
                        ), f"Texture file {tex_file_path} does not exist."

                    tex.attrib["file"] = tex_file_path

                    image = cv2.imread(tex.attrib["file"])

                    if "hsv" in texture_params:
                        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        hsv_ranges = texture_params["hsv"]
                        hsv_range_choice = np.random.choice(range(len(hsv_ranges)))
                        hsv_range = hsv_ranges[hsv_range_choice]
                        hue = np.random.choice(range(hsv_range[0], hsv_range[3] + 1))
                        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180

                        out_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                    else:
                        out_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif texture_params["texture_type"] == "wood":
                    if "table" in obj_name:
                        texture_folder = os.path.join(
                            ASSETS_ROOT, "scenes/mimiclabs_scenes/textures/wood"
                        )
                    else:
                        texture_folder = os.path.join(
                            ASSETS_ROOT, "scenes/mimiclabs_scenes/textures/wood"
                        )
                    texture_files = os.listdir(texture_folder)
                    tex_file = np.random.choice(texture_files)
                    tex.attrib["file"] = os.path.join(texture_folder, tex_file)

                    image = cv2.imread(tex.attrib["file"])

                    if "hsv" in texture_params:
                        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        hsv_ranges = texture_params["hsv"]
                        hue, _, _ = sample_hsv_from_hsv_ranges(hsv_ranges)
                        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180

                        out_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
                    else:
                        out_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif texture_params["texture_type"] == "color":
                    image = cv2.imread(tex_file)
                    H, W = image.shape[:2]

                    hsv_ranges = texture_params["hsv"]
                    hue, sat, val = sample_hsv_from_hsv_ranges(hsv_ranges)

                    # Create solid color texture with sampled HSV values
                    out_hsv = np.stack(
                        [
                            hue * np.ones([H, W]),
                            sat * np.ones([H, W]),
                            val * np.ones([H, W]),
                        ],
                        -1,
                    ).astype(np.uint8)
                    out_rgb = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2RGB)

                elif texture_params["texture_type"] == "fractal":
                    hsv_ranges = texture_params["hsv"]
                    turbulence = texture_params["turbulence"]
                    sigma = texture_params["sigma"]

                    hue, sat, val = sample_hsv_from_hsv_ranges(hsv_ranges)
                    # 170-10 is red, 50-70 is green, 110-130 is blue
                    out_hsv = np.stack(
                        [
                            hue * np.ones([H, W]),
                            sat * np.ones([H, W]),
                            val * np.ones([H, W]),
                        ],
                        -1,
                    ).astype(np.uint8)
                    out_rgb = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2RGB)

                    # iteratively add noise to texture
                    ratio = H
                    while ratio != 1:
                        noise = cv2.resize(
                            np.random.normal(0, sigma, (H // ratio, W // ratio, 3)),
                            dsize=(W, H),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        out_rgb = out_rgb + noise
                        out_rgb = np.clip(out_rgb, 0, 255)
                        ratio = (ratio // turbulence) or 1
                    out_rgb = out_rgb.astype(np.uint8)

                elif texture_params["texture_type"] == "jitter":
                    hsv_ranges = texture_params["hsv"]
                    hue, sat, val = sample_hsv_from_hsv_ranges(hsv_ranges)

                    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(img_hsv)
                    avg_hue, avg_sat, avg_val = np.mean(h), np.mean(s), np.mean(v)
                    h = np.mod(h - avg_hue + hue, 180).astype(np.uint8)
                    s = np.clip(s - avg_sat + sat, 0, 255).astype(np.uint8)
                    v = np.clip(v - avg_val + val, 0, 255).astype(np.uint8)
                    img_hsv = cv2.merge([h, s, v])
                    out_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

                time_str = datetime.datetime.fromtimestamp(time.time()).strftime(
                    "%Y%m%d"
                )
                out_path = (
                    pathlib.Path(MIMICLABS_TMP_FOLDER)
                    / "textures"
                    / time_str
                    / f"texture-{uuid.uuid4()}.png"
                )
                os.makedirs(out_path.parent, exist_ok=True)
                out_path = str(out_path)
                cv2.imwrite(out_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

                tex.attrib["file"] = out_path

    def _setup_placement_initializer(self, mujoco_arena):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.conditional_placement_initializer = SiteSequentialCompositeSampler(
            name="ConditionalSiteSampler"
        )
        self.conditional_placement_on_objects_initializer = SequentialCompositeSampler(
            name="ConditionalObjectSampler"
        )
        self._add_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict()

        for object_name, object_body in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(
                object_body.root_body
            )

        for fixture_name, fixture_body in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(
                fixture_body.root_body
            )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            sensors = []
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked drawer version with rotatable handle

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return (
                T.pose_inv(
                    T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                )
                if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                else np.eye(4)
            )

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        for i, obj in enumerate(self.objects):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(
                obj_name=obj.name, modality="object"
            )

            sensors += obj_sensors
            names += obj_sensor_names

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [
                    name not in obs_cache
                    for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]

        return sensors, names

    def _add_placement_initializer(self):

        mapping_inv = {}
        for k, values in self.parsed_problem["fixtures"].items():
            for v in values:
                mapping_inv[v] = k
        for k, values in self.parsed_problem["objects"].items():
            for v in values:
                mapping_inv[v] = k

        regions = self.parsed_problem["regions"]
        initial_state = self.parsed_problem["initial_state"]
        problem_name = self.parsed_problem["problem_name"]

        conditioned_initial_place_state_on_sites = []
        conditioned_initial_place_state_on_objects = []
        conditioned_initial_place_state_in_objects = []

        for state in initial_state:
            if state[0] == "on" and state[2] in self.objects_dict:
                conditioned_initial_place_state_on_objects.append(state)
                continue

            # (Yifeng) Given that an object needs to have a certain "containing" region in order to hold the relation "In", we assume that users need to specify the containing region of the object already.
            if state[0] == "in" and state[2] in regions:
                conditioned_initial_place_state_in_objects.append(state)
                continue
            # Check if the predicate is in the form of On(object, region)
            if state[0] == "on" and state[2] in regions:
                object_name = state[1]
                region_name = state[2]
                target_name = regions[region_name]["target"]
                x_ranges, y_ranges = rectangle2xyrange(regions[region_name]["ranges"])
                yaw_rotation = regions[region_name]["yaw_rotation"]
                if (
                    target_name in self.objects_dict
                    or target_name in self.fixtures_dict
                ):
                    conditioned_initial_place_state_on_sites.append(state)
                    continue
                if self.is_fixture(object_name):
                    # This is to place environment fixtures.
                    fixture_sampler = MultiRegionRandomSamplerWithYaw(
                        f"{object_name}_sampler",
                        mujoco_objects=self.fixtures_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=yaw_rotation,
                        rotation_axis="z",
                        z_offset=self.z_offset,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(fixture_sampler)
                else:
                    # This is to place movable objects.
                    assert self.objects_dict[object_name].rotation_axis is not None
                    if self.objects_dict[object_name].rotation is not None:
                        # yaw_rotation for objects in object class overrides spec in bddl
                        rotation_kwargs = dict(
                            rotation=self.objects_dict[object_name].rotation,
                            rotation_axis=self.objects_dict[object_name].rotation_axis,
                        )
                    else:
                        assert yaw_rotation is not None
                        rotation_kwargs = dict(
                            rotation=yaw_rotation,
                            rotation_axis=self.objects_dict[object_name].rotation_axis,
                        )
                    region_sampler = get_region_samplers(
                        problem_name, mapping_inv[target_name]
                    )(
                        object_name,
                        self.objects_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        reference_pos=self.workspace_offset,
                        **rotation_kwargs,
                    )
                    self.placement_initializer.append_sampler(region_sampler)
            if state[0] in ["open", "close"]:
                # If "open" is implemented, we assume "close" is also implemented
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "open":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_open_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_close_ranges"
                        ]

                    property_initializer = OpenCloseSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)
            elif state[0] in ["turnon", "turnoff"]:
                # If "turnon" is implemented, we assume "turnoff" is also implemented.
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "turnon":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnon_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnoff_ranges"
                        ]

                    property_initializer = TurnOnOffSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)

        # Place objects that are on sites
        for state in conditioned_initial_place_state_on_sites:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]
            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = SiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )
        # Place objects that are on other objects
        for state in conditioned_initial_place_state_on_objects:
            object_name = state[1]
            other_object_name = state[2]
            sampler = ObjectBasedSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[0.0, 0.0]],
                y_ranges=[[0.0, 0.0]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_on_objects_initializer.append_sampler(
                sampler, {"reference": other_object_name}
            )
        # Place objects inside some containing regions
        for state in conditioned_initial_place_state_in_objects:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]

            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = InSiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                # x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                # y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            for object_property_initializer in self.object_property_initializers:
                if isinstance(object_property_initializer, OpenCloseSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(
                        joint_pos
                    )
                elif isinstance(object_property_initializer, TurnOnOffSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(
                        joint_pos
                    )
                else:
                    print("Warning!!! This sampler doesn't seem to be used")
            # robosuite didn't provide api for this stepping. we manually do this stepping to increase the speed of resetting simulation.
            mujoco.mj_step1(self.sim.model._model, self.sim.data._data)

            object_placements = self.placement_initializer.sample()
            object_placements = self.conditional_placement_initializer.sample(
                self.sim, object_placements
            )
            object_placements = (
                self.conditional_placement_on_objects_initializer.sample(
                    object_placements
                )
            )
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name not in list(self.fixtures_dict.keys()):
                    # This is for movable object resetting (setting free joint)
                    self.sim.data.set_joint_qpos(
                        obj.joints[-1],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                else:
                    # This is for fixture resetting
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): A dictionary containing the state to reset to.
                Contains keys "states" and "model".
        """
        if "model" in state:
            # Edit model xml and reset
            xml = self.edit_model_xml(state["model"])
            self.reset_from_xml_string(xml)

        if "states" in state:
            # Reset to state
            self.sim.reset()
            self.sim.set_state_from_flattened(state["states"])
            self.sim.forward()

        return self._get_observations(force_update=True)

    def reset_from_xml_string(self, xml_string):
        """
        Resets object textures to ones currently in the model before
        calling reset_from_xml_string()
        """
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_string)
        asset = root.find("asset")

        # Resetting table texture to default
        tex = asset.find("./texture[@name='tex-table']")
        if tex is not None:
            # Find original texture file from original xml
            if not os.path.exists(tex.attrib["file"]):
                orig_scene = ET.parse(self._arena_xml)
                orig_tex_file = orig_scene.find(
                    "./asset/texture[@name='tex-table']"
                ).attrib["file"]
                orig_tex_file = os.path.join(
                    os.path.dirname(self._arena_xml), orig_tex_file
                )
                tex.attrib["file"] = orig_tex_file

        # Resetting all object textures to default
        for _, obj in self.objects_dict.items():
            objtex = obj.asset.find("./texture")
            if objtex is not None:
                texname = objtex.attrib["name"]
                tex = asset.find(f"./texture[@name='{texname}']")
                if not os.path.exists(tex.attrib["file"]):
                    tex.attrib["file"] = objtex.attrib["file"]

        modified_xml_string = ET.tostring(root, encoding="utf8").decode("utf8")
        super().reset_from_xml_string(modified_xml_string)

    def _check_success(self):
        """
        This needs to match with the goal description from the bddl file

        Returns:
            bool: True if drawer has been opened
        """
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4 and len(action) > 4:
            # Convert OSC_POSITION action
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)

        obs, reward, done, info = super().step(action)
        done = self._check_success()

        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        self._post_process()

        return reward, done, info

    def _post_process(self):
        # Update some object states, such as light switching etc.
        for object_state in self.tracking_object_states_change:
            object_state.update_state()

    def get_robot_state_vector(self, obs):
        return np.concatenate(
            [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
        )

    def is_fixture(self, object_name):
        """
        Check if an object is defined as a fixture in the task

        Args:
            object_name (str): The name string of the object in query
        """
        return object_name in list(self.fixtures_dict.keys())

    @property
    def language_instruction(self):
        return self.parsed_problem["language"]

    def get_object(self, object_name):
        for query_dict in [
            self.fixtures_dict,
            self.objects_dict,
            self.object_sites_dict,
        ]:
            if object_name in query_dict:
                return query_dict[object_name]
