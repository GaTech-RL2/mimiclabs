"""
Script to add observations to mimiclabs datasets.

Example usage:
    python scripts/add_obs_to_mimiclabs_datasets.py \
        --input_root_dir /path/to/input_dir \
        --output_root_dir /path/to/output_dir

    python scripts/add_obs_to_mimiclabs_datasets.py \
        --input_root_dir ../../datasets/mimiclabs_study \
        --output_root_dir ../../datasets/mimiclabs_study
"""

import os
import argparse


def generate_shell_script(args):
    output_file = os.path.join(
        os.path.dirname(__file__), "add_obs_to_mimiclabs_datasets.sh"
    )

    spacing = " " * 4
    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")

        labs = os.listdir(args.input_root_dir)
        for lab in labs:
            if not os.path.isdir(os.path.join(args.input_root_dir, lab)):
                continue
            tasks = os.listdir(os.path.join(args.input_root_dir, lab))
            for task in tasks:
                if not os.path.isdir(os.path.join(args.input_root_dir, lab, task)):
                    continue
                task_variants = os.listdir(os.path.join(args.input_root_dir, lab, task))
                for task_variant in task_variants:
                    task_variant_dir = os.path.join(
                        args.input_root_dir, lab, task, task_variant
                    )
                    if not os.path.isdir(task_variant_dir):
                        continue
                    # Check if demo.hdf5 exists
                    if not os.path.exists(os.path.join(task_variant_dir, "demo.hdf5")):
                        print(
                            f"Skipping {task_variant_dir} because demo.hdf5 does not exist"
                        )
                        continue

                    f.write(f"python scripts/dataset_states_to_obs.py \\\n")
                    f.write(f"{spacing}--dataset {task_variant_dir}/demo.hdf5 \\\n")
                    f.write(f"{spacing}--output_dir {task_variant_dir} \\\n")
                    f.write(
                        f"{spacing}--camera_names agentview robot0_eye_in_hand \\\n"
                    )
                    if args.add_depth_obs:
                        f.write(f"{spacing}--camera_depths \\\n")
                    f.write(f"{spacing}--camera_height 128 \\\n")
                    f.write(f"{spacing}--camera_width 128 \n")
                    f.write("\n")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root_dir",
        type=str,
        required=True,
        help="directory to read dataset from",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        required=True,
        help="directory to write converted dataset to",
    )
    parser.add_argument(
        "--add_depth_obs",
        action="store_true",
        help="whether to add camera depths to the dataset",
    )

    args = parser.parse_args()

    script_path = generate_shell_script(args)

    print(
        "Generated shell script to add observations to mimiclabs datasets at",
        script_path,
    )
