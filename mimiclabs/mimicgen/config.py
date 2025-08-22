import os
from pathlib import Path

import mimicgen
from mimicgen.configs.config import ConfigMeta, MG_Config
from robomimic.config.config import Config

import mimiclabs
import mimiclabs.mimiclabs.envs.bddl_utils as BDDLUtils
from mimiclabs.mimicgen.env_interface import MG_MimicLabs


def generate_mg_config_classes(
    task_suite_name,
    camera_names,
    camera_height,
    camera_width,
):
    """
    Generates MimicGen config classes for all BDDL files under the specified task suite name.
    This function dynamically creates config classes that inherit from `MG_Config` and
    registers them with the `ConfigMeta` metaclass. The "task_config" and "obs_config" methods
    are populated based on the parsed BDDL files and args to this function respectively, allowing
    for flexible task and observation configurations.

    Args:
        task_suite_name (str): The name of the task suite, which corresponds to a directory
            containing BDDL files.
        camera_names (list): List of camera names to be used for observations.
        camera_height (int): Height of the camera images.
        camera_width (int): Width of the camera images.
    """
    bddl_base_path = os.path.join(mimiclabs.__path__[0], "mimiclabs", "task_suites")
    bddl_suite_path = os.path.join(bddl_base_path, task_suite_name)
    for task_bddl_file in os.listdir(bddl_suite_path):
        if not task_bddl_file.endswith(".bddl"):
            continue
        bddl_file_name = os.path.join(bddl_suite_path, task_bddl_file)
        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file_name)
        
        def make_task_config(problem):
            """Factory function to create task_config with proper closure"""
            def task_config(self):
                """
                This function populates the `config.task` attribute of the config,
                which has settings for each object-centric subtask in a task.
                """
                num_subtasks = len(problem["demonstration_states"])
                for i, subtask_predicate in enumerate(
                    problem["demonstration_states"]
                ):
                    subtask_id = f"subtask_{i+1}"
                    selection_strategy_kwargs = dict(nn_k=3)
                    object_ref = None
                    for obj_name in problem["obj_of_interest"]:
                        # Currently, we assume that the reference object is the
                        # last entity in the subtask predicate that starts with the object name.
                        # This is a heuristic and may need to be adjusted based on the demonstration predicate.
                        if subtask_predicate[-1].startswith(obj_name):
                            object_ref = obj_name
                            break
                    if i == num_subtasks - 1:
                        subtask_term_signal = None
                        subtask_term_offset_range = None
                    else:
                        subtask_term_signal = (
                            MG_MimicLabs._get_term_signal_key_from_demo_state(
                                subtask_predicate, prefix=subtask_id
                            )
                        )
                        subtask_term_offset_range = (
                            10,
                            20,
                        )  # AM: could be low (0,0) if demo states are precise
                    self.task.task_spec[subtask_id] = dict(
                        object_ref=object_ref,
                        subtask_term_signal=subtask_term_signal,
                        subtask_term_offset_range=subtask_term_offset_range,
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs=selection_strategy_kwargs,
                        action_noise=0.01,  # adds noise to actions
                        num_interpolation_steps=5,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    )
                self.task.task_spec.do_not_lock_keys()
            return task_config

        def make_obs_config(cam_names, cam_height, cam_width):
            """Factory function to create obs_config with proper closure"""
            def obs_config(self):
                """
                This function populates the `config.obs` attribute of the config, which has
                settings for which observations to collect during data generation.
                """
                self.obs.collect_obs = True
                self.obs.camera_names = cam_names
                self.obs.camera_height = cam_height
                self.obs.camera_width = cam_width
            return obs_config

        cls_name = Path(task_bddl_file).stem
        bases = (MG_Config,)
        attr_dict = {
            "NAME": cls_name,
            "TYPE": task_suite_name,
            "task_config": make_task_config(parsed_problem),
            "obs_config": make_obs_config(camera_names, camera_height, camera_width),
        }
        # create and register config class
        ConfigMeta(cls_name, bases, attr_dict)
