import re

from ...utils import disable_module_import

with disable_module_import("libero", "libero", "envs"):
    from libero.libero.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT
    from libero.libero.envs.objects.target_zones import *

from .articulated_objects import *
from .libero_objects import *
from .mimicgen_objects import *
from .objaverse_objects import *
from .primitive_objects import *

try:
    from .robocasa_objects import *
except ImportError:
    print("WARNING: could not import robocasa objects")


def get_object_fn(category_name):
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
