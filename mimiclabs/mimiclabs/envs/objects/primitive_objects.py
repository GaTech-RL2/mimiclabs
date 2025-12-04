import numpy as np
import re

from robosuite.models.objects.primitive.box import BoxObject

from libero.libero.envs.base_object import register_object


@register_object
class MujocoCuboid(BoxObject):
    """
    A wrapper around BoxObject that follows the naming convention used in objaverse objects.

    Args:
        name (str): Name of the object instance
        obj_name (str): Object type name (used for category naming)
        joints (list): List of joint specifications. Defaults to a free joint with damping.
    """

    def __init__(
        self,
        name="mujoco_cuboid",
        obj_name="cuboid",
        joints=[dict(type="free", damping="0.0005")],
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        friction=None,
        rgba=None,
        solref=None,
        solimp=None,
        material=None,
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        # Convert joints list to the format expected by BoxObject
        # BoxObject expects joints="default" or a list of joint dicts
        if joints is None:
            joints_param = None
        elif len(joints) == 0:
            joints_param = None
        else:
            joints_param = joints

        super().__init__(
            name=name,
            size=size,
            size_max=size_max,
            size_min=size_min,
            density=density,
            friction=friction,
            rgba=rgba,
            solref=solref,
            solimp=solimp,
            material=material,
            joints=joints_param,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

        # Add category name.
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}
