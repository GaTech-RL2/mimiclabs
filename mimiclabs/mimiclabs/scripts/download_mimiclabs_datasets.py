"""
Script to download datasets for the MimicLabs study.

Example usage:
    python scripts/download_mimiclabs_datasets.py --download_dir ../../datasets --labs lab1 lab2 --dry_run

"""

import os
import shutil
import argparse

from huggingface_hub import HfFileSystem, hf_hub_download


REPO_ID = "vaibhavsaxena11/mimiclabs_datasets"
MIMICLABS_DATASETS = {
    "lab1": "mimiclabs_study/lab1",
    "lab2": "mimiclabs_study/lab2",
    "lab3": "mimiclabs_study/lab3",
    "lab4": "mimiclabs_study/lab4",
    "lab5": "mimiclabs_study/lab5",
    "lab6": "mimiclabs_study/lab6",
    "lab7": "mimiclabs_study/lab7",
    "lab8": "mimiclabs_study/lab8",
}


def download_folder_from_hf(
    hf_folder_path,
    download_dir,
):
    """
    Function downloading all contents of folder hf_folder_path into download_dir.
    This function is recursive, so it will download all subfolders and files in the folder.

    Args:
        hf_folder_path (str): Path to the folder in HuggingFace, e.g. "datasets/username/repo_name/folder_name".
        download_dir (str): Directory to download the folder to.
    """
    fs = HfFileSystem()

    # List all files/folders in the repo folder
    files = fs.ls(hf_folder_path, detail=True)

    for file in files:
        filename = file[
            "name"
        ]  # e.g. "datasets/username/repo_name/folder_name/file.txt"
        basename = os.path.basename(filename)  # name of the file/folder

        if file["type"] == "directory":
            # Recursively download folder
            download_subdir = os.path.join(download_dir, basename)
            os.makedirs(download_subdir, exist_ok=True)
            download_folder_from_hf(
                hf_folder_path=filename,
                download_dir=download_subdir,
            )
        else:
            # Download file
            tmp_dir = os.path.join(download_dir, "tmp")
            filename_in_repo = filename[len(f"datasets/{REPO_ID}/") :]
            fpath = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename_in_repo,
                repo_type="dataset",
                cache_dir=tmp_dir,
            )
            # Move downloaded file to the download directory (remember, fpath is a pointer, and actual file has a different name)
            shutil.move(os.path.realpath(fpath), os.path.join(download_dir, basename))
            shutil.rmtree(tmp_dir)


def download_datasets(args):
    """
    Download datasets for the specified labs.
    """

    if "all" in args.labs:
        # If 'all' is specified, download all datasets
        assert (
            len(args.labs) == 1
        ), "If 'all' is specified, no other labs should be specified."
        labs = MIMICLABS_DATASETS.keys()
    else:
        labs = args.labs
        assert all(
            lab in MIMICLABS_DATASETS for lab in labs
        ), "Some specified labs are not available in the datasets."

    # Download datasets for the specified labs
    for lab in labs:
        dataset_path = os.path.join(args.download_dir, MIMICLABS_DATASETS[lab])

        if args.dry_run:
            print(
                f"Dataset at {REPO_ID}/{MIMICLABS_DATASETS[lab]} will be downloaded to {dataset_path}"
            )
        else:
            print(f"Downloading {REPO_ID}/{MIMICLABS_DATASETS[lab]} to {dataset_path}")

            # Check if download_dir exists, if not create it
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path, exist_ok=True)

            download_folder_from_hf(
                hf_folder_path=f"datasets/{REPO_ID}/{MIMICLABS_DATASETS[lab]}",
                download_dir=dataset_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--download_dir",
        type=str,
        required=True,
        help="directory to download datasets to.",
    )

    # dataset type to download datasets for
    parser.add_argument(
        "--labs",
        type=str,
        nargs="+",
        default=["all"],
        help="labs to download datasets for. Defaults to all labs. Pass 'all' to download all labs.",
    )

    # dry run - don't actually download datasets, but print which datasets would be downloaded
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="set this flag to do a dry run to only print which datasets would be downloaded",
    )

    args = parser.parse_args()
    download_datasets(args)
