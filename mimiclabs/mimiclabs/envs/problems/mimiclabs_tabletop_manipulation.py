"""
This file contains the MimicLabs Tabletop Manipulation classes for 8 different scenes (adapted from LIBERO).
"""

from robosuite.utils.mjcf_utils import new_site

import libero

from ..bddl_base_domain import BDDLBaseDomain
from ..bddl_base_domain import register_problem
from ..objects import *
from ..robots import *
from ..predicates import *
from ..regions import *


class MimicLabs_Tabletop_Manipulation_Base(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        elif hasattr(self, "table_full_size") is False:
            self.table_full_size = (1.0, 1.2, 0.05)

        if hasattr(self, "table_offset") is False:
            self.table_offset = (0.0, 0, 0.90)

        # For z offset of environment fixtures
        if hasattr(self, "z_offset") is False:
            self.z_offset = 0.01 - self.table_full_size[2]
        if "workspace_offset" not in kwargs or kwargs["workspace_offset"] is None:
            kwargs.update({"workspace_offset": self.table_offset})
        if "arena_type" not in kwargs or kwargs["arena_type"] is None:
            kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {
                    "scene_xml": f"{libero.__path__[0]}/libero/assets/scenes/libero_kitchen_tabletop_base_style.xml"
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if "table" in fixture_category:
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena, object_params):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name, **object_params.get(object_name, {})
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2 + self.workspace_offset[0],
                    (ranges[3] + ranges[1]) / 2 + self.workspace_offset[1],
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    z_offset=self.workspace_offset[2],
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos + np.array([0.0, 0.0, -0.90]),
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for name, body in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # Handling composite objects
                            parts = [body.get_obj(), *body.get_obj().findall(".//body")]
                        else:
                            parts = body.worldbody.find("body").findall(".//body")
                    except:
                        continue

                    for part in parts:
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")  # joints within the part
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,  # name in bddl
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
                                # NOTE(VS) this creates SiteObject's with empty list of joints for the "object" part
                                # and finally overrides them with the SiteObject with the correct list of joints upon
                                # reaching the correct part that has the required site as well as the joint.
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_conj = self.parsed_problem["goal_state"][0]
        goal_state = self.parsed_problem["goal_state"][1:]
        if goal_conj == "and":
            result = True
            for state in goal_state:
                result = self._eval_predicate(state) and result
        elif goal_conj == "or":
            result = False
            for state in goal_state:
                result = self._eval_predicate(state) or result
        else:
            raise ValueError(f"Unsupported goal conjunction {goal_conj}.")
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )


@register_problem
class MimicLabs_Lab1_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {
                    "scene_xml": f"{libero.__path__[0]}/libero/assets/scenes/libero_kitchen_tabletop_base_style.xml"
                }
            )
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "gray-ceramic",
                        "wall_style": "yellow-linen",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab2_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):
    def __init__(self, bddl_file_name, *args, **kwargs):
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab2.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "wood-plank",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab3_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab3.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "wall_style": "wall0",
                        "floor_style": "carpet",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab4_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab4.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "wall_style": "yellow-linen",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab5_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab5.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "wall_style": "wall8",
                        "floor_style": "carpet",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab6_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab6.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "wall_style": "wall1",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab7_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab7.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "wall_style": "wall2",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)


@register_problem
class MimicLabs_Lab8_Tabletop_Manipulation(MimicLabs_Tabletop_Manipulation_Base):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/mimiclabs_scenes/lab8.xml"})

        super().__init__(bddl_file_name, *args, **kwargs)
