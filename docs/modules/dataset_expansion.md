# Dataset Expansion using MimicGen

<!-- MimicGen intro -->

## Preparing your source demonstrations

After collecting your demonstrations, run the following script to add subtask information to the dataset:

```bash
$ cd <PATH_TO_THIS_REPO>/mimiclabs/mimicgen
$ python scripts/prepare_src_dataset.py --dataset </PATH/TO/DATASET/....hdf5> \
```
This script defaults to using the `MG_MimicLabs` MimicGen interface for injecting object-centric information into the demo file, which is subsequently needed for the dataset generation. If you wish to use your own interface, make sure to import it in `scripts/prepare_src_dataset.py` and pass the interface name using the `--env_interface` flag.

## Creating MimicGen configs and generation jobs

The MimicGen data generation pipeline requires a set of configurations for each task created by the user as a BDDL. We provide scripts to generate these configs directly from task BDDLs, and subsequently launch data generation jobs. Below is an example:

```bash
$ cd <PATH_TO_THIS_REPO>/mimiclabs/mimicgen
$ python scripts/generate_configs_and_jobs.py \
    --task_suite_name <YOUR/TASK/SUITE> \
    --source_demos_dir <DIR/TO/SOURCE/DEMOS> \
    --generation_dir <DIR/TO/GENERATION/DEMOS> \
    --num_demos <NUM_DEMOS_PER_TASK>
```
Please see `mimiclabs/mimicgen/scripts/generate_configs_and_jobs.py` for additional config parameters that affect the config generation process.

The above script will generate config templates for MimicGen as JSON files under `<PATH_TO_THIS_REPO>/mimiclabs/mimicgen/exps/<YOUR/TASK/SUITE>`. It also exports a bash script for running all data generation jobs. You can run them in parallel using:
```bash
$ ./exps/<YOUR/TASK/SUITE>/jobs.sh
```

Each data data generation job runs `mimiclabs/mimicgen/scripts/generate_dataset.py`.

### Re-using source demonstrations across a task suite

To re-use source datasets for new BDDLs when using MimicGen, create a `task_suite` containing all the new BDDLs, then run:

```bash
$ cd <PATH_TO_THIS_REPO>/mimiclabs/mimicgen
$ python scripts/generate_configs_and_jobs.py \
    --task_suite_name <YOUR/TASK/SUITE> \
    --source_dataset_path <PATH/TO/REUSABLE/SOURCE/DEMOS> \
    --generation_dir <DIR/TO/GENERATION/DEMOS> \
    --num_demos <NUM_DEMOS_PER_TASK>
```
where you can provide the source demos that you want to re-use for all tasks in the suite, in the `--source_dataset_path` arg.
