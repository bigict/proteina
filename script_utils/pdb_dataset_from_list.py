# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import contextlib
from dotenv import load_dotenv
import hydra
from loguru import logger

@contextlib.contextmanager
def _open_file(fname):
    if fname == "-":
        yield sys.stdin
    else:
        with open(fname, "r") as f:
            yield f

def _read_pid(f):
    for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
        if not line.startswith("#"):
            yield line

def main(args):
    load_dotenv()

    # Load experiment config
    config_path = "../configs/experiment_config"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
        logger.info(f"Exp config {cfg_exp}")
    
    # Load data config
    dataset_config_subdir = cfg_exp.get("dataset_config_subdir", None)
    if dataset_config_subdir is not None:
        # if args.dataset_config_subdir:
        config_path = f"../configs/datasets_config/{dataset_config_subdir}"
    else:
        config_path = "../configs/datasets_config/"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"])
        logger.info(f"Data config {cfg_data}")

    # create datamodule containing default train and val dataloader
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    assert datamodule.dataselector is not None
    df_data = datamodule.dataselector.create_dataset()
    # filter by pdb_id
    if args.pid_list_file:
        with _open_file(args.pid_list_file) as f:
            pid_list = list(_read_pid(f))
        df_data = df_data.loc[df_data["pdb"].isin(pid_list)]
        logger.info(f"{len(df_data)} chains remaining")

    logger.info(
        f"Dataset created with {len(df_data)} entries. Now downloading structure data..."
    )

    datamodule._download_structure_data(df_data["pdb"].tolist())

    # process pdb files into seperate chains and save processed objects as .pt files
    datamodule._process_structure_data(
        df_data["pdb"].tolist(), df_data["chain"].tolist()
    )
    # save df_data to disk for later use (in splitting, dataloading etc)
    file_identifier = datamodule._get_file_identifier(datamodule.dataselector)
    df_data_name = f"{file_identifier}.csv"
    logger.info(f"Saving dataset csv to {df_data_name}")
    df_data.to_csv(datamodule.dataselector.data_dir / df_data_name, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="training_ca",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--pid_list_file",
        type=str,
        default=None,
        help="Filter pdb id by list file.",
    )
    
    args = parser.parse_args()

    main(args)
