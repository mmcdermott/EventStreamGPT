#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="EventStreamGPT",
    version="0.0.1",
    description="EventStreamGPT",
    author="Matthew McDermott",
    author_email="mattmcdermott8@gmail.com",
    url="https://github.com/mmcdermott/EventStreamGPT",
    install_requires=[
        "pytorch-lightning",
        "hydra-core",
        "transformers",
        "torchmetrics",
        "wandb",
        "ml-mixins",
        "pytorch_lognormal_mixture",
    ],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    scripts=[
        "scripts/pretrain.py",
        "scripts/finetune.py",
        "scripts/get_embeddings.py",
        "scripts/launch_wandb_hp_sweep.py",
    ],
)
