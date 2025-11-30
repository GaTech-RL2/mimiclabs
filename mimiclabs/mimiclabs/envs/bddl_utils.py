import os
import numpy as np
import json
from bddl.parsing import *

from ..utils import disable_module_import

with disable_module_import("libero", "libero", "envs"):
    from libero.libero.envs.bddl_utils import get_problem_info, get_regions, get_scenes

import mimiclabs


def package_predicates(group, goals, name, part):
    if not isinstance(group, list):
        raise Exception("Error with " + name + part)
    if group[0] in ["and", "or"]:
        goals.append(group.pop(0))
    else:
        goals.append("and")
        group = [group]
    for predicate in group:
        goals.append(predicate)


def get_textures(group):
    textures = {}
    for texgrp in group[1:]:
        obj_name = texgrp[0]
        textures[obj_name] = {}
        for prop in texgrp[1:]:
            if prop[0] == ":type":
                textures[obj_name]["texture_type"] = prop[1]
            elif prop[0] == ":hsv":
                vals_list = prop[1]
                textures[obj_name]["hsv"] = [
                    [eval(val) for val in vals] for vals in vals_list
                ]
                # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
            elif prop[0] == ":turbulence":
                # determines how quickly low freq noise is replaced by high freq noise
                textures[obj_name]["turbulence"] = eval(prop[1])
            elif prop[0] == ":sigma":
                # stddev of added noise
                textures[obj_name]["sigma"] = eval(prop[1])
            elif prop[0] == ":files":
                files_list = prop[1]
                textures[obj_name]["files"] = files_list

    return textures


def get_camera_poses(group):
    camera = {}
    camera["ranges"] = []
    camera["jitter_mode"] = "uniform"
    camera["unit"] = "radians"
    for subgrp in group[1:]:
        if subgrp[0] == ":ranges":
            for cam_range in subgrp[1]:
                # ranges in spherical coordinates (physics convention)
                camera["ranges"].append([eval(val) for val in cam_range])
        if subgrp[0] == ":jitter_mode":
            camera["jitter_mode"] = subgrp[1]
        if subgrp[0] == ":unit":
            camera["unit"] = subgrp[1]
    assert camera["jitter_mode"] in [
        "uniform",
        "normal",
    ], f"Camera jitter mode {camera['jitter_mode']} not supported."
    assert camera["unit"] in [
        "radians",
        "degrees",
    ], f"Camera pose unit should be radians or degrees, {camera['unit']} not supported."
    return camera


def get_table_params(group):
    """Parse table parameters (size as [width, length, height]) from BDDL."""
    table_params = {}
    for subgrp in group[1:]:
        if subgrp[0] == ":size":
            # Parse size as a list of 3 values: width, length, height
            size_values = [eval(val) for val in subgrp[1:]]
            table_params["size"] = size_values
    return table_params


def get_object_params(group):
    """
    Parse object parameters from BDDL generically.
    Returns a dict mapping object names to their parameters as kwargs.
    """
    object_params = {}
    for obj_group in group[1:]:
        # obj_group[0] is like :object_1, strip the colon to get object name
        obj_name = obj_group[0].lstrip(":")
        object_params[obj_name] = {}

        for param in obj_group[1:]:
            # Strip the colon from parameter name
            param_name = param[0].lstrip(":")

            # Evaluate all parameter values
            param_values = [eval(val) for val in param[1:]]

            # If single value, unwrap from list; otherwise keep as list
            if len(param_values) == 1:
                object_params[obj_name][param_name] = param_values[0]
            else:
                object_params[obj_name][param_name] = param_values

    return object_params


def robosuite_parse_problem(problem_filename):
    if problem_filename.endswith(".json"):
        parsed_problem = json.load(open(problem_filename, "r"))
        return parsed_problem

    domain_name = "robosuite"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        textures = {}
        camera = {}
        lighting = {}
        table_params = {}
        object_params = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        demonstration_states = []
        fixtures = {}
        regions = {}
        scene_properties = {}
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":textures":
                textures = get_textures(group)
            elif t == ":camera":
                camera = get_camera_poses(group)
            elif t == ":table":
                table_params = get_table_params(group)
            elif t == ":object_params":
                object_params = get_object_params(group)
            elif t == ":lighting":
                lighting["source"] = []
                for subgrp in group[1:]:
                    if subgrp[0] == ":shadow":
                        lighting["shadow"] = True if subgrp[1] == "true" else False
                    elif subgrp[0] == ":source":
                        # position of light source pointing at the origin
                        # ranges in spherical coordinates (physics convention)
                        for src_range in subgrp[1]:
                            lighting["source"].append([eval(val) for val in src_range])
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group

            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            elif t == ":demonstration":
                group.pop(0)
                demonstration_states = group
            else:
                print("%s is not recognized in problem" % t)
        return {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "textures": textures,
            "camera": camera,
            "table_params": table_params,
            "object_params": object_params,
            "lighting": lighting,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "demonstration_states": demonstration_states,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
        }
    else:
        raise Exception(
            f"Problem {behavior_activity} {activity_definition} does not match problem pattern"
        )


def resolve_bddl_file_name(bddl_file_name):
    """Try to replace BDDL file path to be relative to MimicLabs installation."""

    bddl_file_name_split = bddl_file_name.split("/")

    check_lst = [
        loc for loc, val in enumerate(bddl_file_name_split) if val == "mimiclabs"
    ]
    if len(check_lst) > 0:
        ind = max(check_lst)  # last occurrence index
        new_path_split = (
            mimiclabs.__path__[0].split("/")
            + ["mimiclabs"]
            + bddl_file_name_split[ind + 1 :]
        )
        new_path = "/".join(new_path_split)
        assert os.path.exists(new_path)
        return new_path
