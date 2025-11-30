# MimicLabs

<p align="center">
  <img width="95.0%" src="./docs/images/mimiclabs-logo-video.gif">
</p>

Welcome to MimicLabs, your one-stop place for collecting and generating datasets for table-top manipulation! MimicLabs provides a framework for describing a suite of MuJoCo-based tasks, collecting expert demonstrations, and large-scale data generation using MimicGen.

This is also the official repository for the study paper ''What Matters in Learning from Large-Scale Datasets for Robot Manipulation'' appearing at [ICLR 2025](https://iclr.cc/).

In this repo, we leverage various open-source projects, including Robosuite, LIBERO, RoboCasa, and MimicGen. We thank their authors for making their code publicly available.

Website: https://robo-mimiclabs.github.io

Paper: https://arxiv.org/abs/2506.13536

Documentation: https://robo-mimiclabs.github.io/docs/getting_started/welcome.html

## Getting started with MimicLabs

MimicLabs allows you to create a suite of tasks, collect demonstrations, and expand your datasets using MimicGen. Our workflow consists of the following 3 stages:

1. Set up your task configs (BDDLs) (use `mimiclabs/mimiclabs`)

2. Collect source demonstrations (use `mimiclabs/data_collection`)

3. Expand your datasets (using MimicGen) (use `mimiclabs/mimicgen`)

We provide detailed documentation for each of these stages under `docs/modules`, including an example workflow in `docs/examples`. For more details, see the [documentation](https://robo-mimiclabs.github.io/docs/getting_started/welcome.html).

For the MimicLabs study, we constructed a vast suite of task configs that you can find in this repo under `mimiclabs/mimiclabs/task_suites/mimiclabs_study`. We also make our simulation datasets available on [Hugging Face](https://huggingface.co/datasets/vaibhavsaxena11/mimiclabs_datasets/tree/main).

## Installation

### 1. Setting up your conda environment
```bash
$ conda create -n mimiclabs python=3.10
$ conda activate mimiclabs
```

### 2. Installing required libraries
Run the following commands to install **Robosuite**, **LIBERO**, **MimicGen**, and **RoboCasa**.
```bash
# install LIBERO
(mimiclabs)$ git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
(mimiclabs)$ cd LIBERO
(mimiclabs)$ pip install -e .
(mimiclabs)$ cd ..

# install MimicGen
(mimiclabs)$ git clone https://github.com/NVlabs/mimicgen.git
(mimiclabs)$ cd mimicgen
(mimiclabs)$ pip install -e .
(mimiclabs)$ cd ..

# (optional) install RoboCasa (for additional assets)
(mimiclabs)$ git clone https://github.com/robocasa/robocasa.git
(mimiclabs)$ cd robocasa
(mimiclabs)$ pip install -e .
# next: follow instructions on their github to download robocasa assets

# install Robomimic
(mimiclabs)$ git clone https://github.com/ARISE-Initiative/robomimic.git
(mimiclabs)$ cd robomimic
(mimiclabs)$ pip install -e .
(mimiclabs)$ cd ..

# install Robosuite
(mimiclabs)$ git clone https://github.com/ARISE-Initiative/robosuite.git
(mimiclabs)$ cd robosuite && git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
(mimiclabs)$ pip install -e .
(mimiclabs)$ cd ..
```

### 3. Installing MimicLabs
```bash
(mimiclabs)$ git clone <link_to_this_repo>
(mimiclabs)$ cd mimiclabs
(mimiclabs)$ pip install -e .
(mimiclabs)$ pip install -r requirements.txt
```

### 4. Downloading MimicLabs assets:
```bash
(mimiclabs)$ cd mimiclabs/mimiclabs
(mimiclabs)$ python scripts/download_mimiclabs_assets.py
```

## Citation

If you find this repo useful, please cite in your work:
```bibtex
@inproceedings{
saxena2025mimiclabs,
    title={What Matters in Learning from Large-Scale Datasets for Robot Manipulation},
    author={Vaibhav Saxena and Matthew Bronars and Nadun Ranawaka Arachchige and Kuancheng Wang and Woo Chul Shin and Soroush Nasiriany and Ajay Mandlekar and Danfei Xu},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://arxiv.org/pdf/2506.13536}
}
```