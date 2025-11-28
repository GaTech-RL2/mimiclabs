# Working with MimicGen in MimicLabs

We provide a detailed description of how we automate integration with MimicGen, and how you can tune it for your work.

Some important scripts are:

1. Preparing source demonstrations: `mimiclabs/mimicgen/scripts/prepare_src_dataset.py`
2. Creating datagen configs: `mimiclabs/mimicgen/scripts/generate_configs_and_jobs.py`
3. Generating datasets: `mimiclabs/mimicgen/scripts/generate_dataset.py`


## Preparing your source demonstrations

The `prepare_src_dataset.py` script is used to add object-centric information to source demonstrations before they can be used for dataset expansion. Below are all paramaeters that can be changed in the this script.

| Argument | Description |
|:--------------------|:--------------------|
| dataset | path to source hdf5 dataset (after post-processing), which will be modified in-place |
| env_interface | interface class to use for this source dataset. default is `MG_MimicLabs`. We built this class to extract object-centric info about demos using `demonstration` states in BDDL |
| env_interface_type | type of environment interface; we set it to `robosuite` by default and is provided by MimicGen |
| n | only process that many trajectories (used for debugging) |
| filter_key | filters the source demos to be `source_hdf5_file[f"mask/{filter_key}"]` |
| output | path to output hdf5; if not specified, dataset is modified in place |


### The MG_MimicLabs interface class

Preparing source demonstrations relies on a MimicGen environment interface that we define for all tasks described in MimicLabs, called `MG_MimicLabs`. It is a subclass of the [RobosuiteInterface](https://github.com/NVlabs/mimicgen/blob/4c7b46e6a912c49cf9072e4c0f873e1aadd42b24/mimicgen/env_interfaces/robosuite.py#L17) class in MimicGen, and overrides its `get_object_poses()` and `get_subtask_term_signals()` functions. We define this interface class in `mimiclabs/mimicgen/env_interface.py`. 

This class uses the `demonstration` states in the BDDL to convert them into subtask keys that are added to `source_hdf5_file[f"data/demo_i/datagen_info"]`. For example, for the given `demonstration` states,
```
(:demonstration
    (:Open wooden_cabinet_1_top_region)
    (:Grasp object_1)
    (:In object_1 wooden_cabinet_1_top_region)
)
```
the following key-value pairs are added for each timestep
```
"subtask_1_open_wooden_cabinet_1_top_region": eval("open", "wooden_cabinet_1_top_region")
"subtask_2_grasp_object_1": eval("grasp", "object_1")
"subtask_3_object_1_wooden_cabinet_1_top_region": eval("in", "object_1", "wooden_cabinet_1_top_region")
```
where `eval()` is a function that evaluates any predicate in the scene. Implementations for different predicate can be found in `mimiclabs/mimiclabs/envs/predicates/predicates.py` which imports multiple implementations from LIBERO.

## Creating MimicGen configs and generation jobs

The `generate_configs_and_jobs.py` is used to create configs for data generation with MimicGen. 

This script calls the `generate_mg_config_classes()` function in `mimiclabs/mimicgen/config.py` to generate config classes. The name of each class is the BDDL filename, with a `TYPE` attribute assigned to be the task suite name. This function dynamically creates config classes that inherit from [MG_Config](https://github.com/NVlabs/mimicgen/blob/4c7b46e6a912c49cf9072e4c0f873e1aadd42b24/mimicgen/configs/config.py#L52) and registers them with the [ConfigMeta](https://github.com/NVlabs/mimicgen/blob/4c7b46e6a912c49cf9072e4c0f873e1aadd42b24/mimicgen/configs/config.py#L37) metaclass. The `task_config()` and `obs_config()` methods are populated based on the parsed BDDL files and args to this function respectively, allowing for flexible task and observation configurations. 

Below are some useful arguments to this script:

| Argument | Description |
|:--------------------|:--------------------|
| task_suite_name | name of task suite to generate demos for |
| source_demos_dir | directory containing source demos for the task suite, one subdirectory per task bddl file |
| source_dataset_path | if provided, uses this path to load source demos for each task instead of looking for demos in source_demos_dir/task_name |
| generation_dir | directory to store generated demos for the task suite, one subdirectory per task bddl file |
| num_demos | number of demos to generate for each task bddl |
| camera_names | list of camera names to use for rendering and store to dataset; default is ["agentview", "robot0_eye_in_hand"] |
| camera_height | camera height to use for rendering; default is 84 |
| camera_width | camera width to use for rendering; default is 84 |


## Generating datasets

The `generate_dataset.py` is finally used to start data generation using MimicGen given a generated config. An important thing to note is that we call the `generate_mg_config_classes()` from `mimiclabs/mimicgen/config.py` in this script to create and register config classes for each BDDL in the task suite. 

Below are some useful arguments to this script:

| Argument | Description |
|:--------------------|:--------------------|
| config | path to MimicGen config json |
| render | pass this to render demo collection playback to screen |
