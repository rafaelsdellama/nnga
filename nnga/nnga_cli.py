""" Command line inferface for package Neural Network Genetic Algorithm
https://medium.com/@trstringer/the-easy-and-nice-way-to-do-cli-apps-in-python-5d9964dc950d
"""
import argparse
import os
from pathlib import Path
import traceback

from nnga.configs import cfg, export_config
from nnga.utils.logger import setup_logger
from nnga.datasets.image_dataset import ImageDataset
from nnga.datasets.csv_dataset import CSVDataset
from nnga.genetic_algorithm.ga import GA


def train(cfg, logger):
    """
    Create datasets, Models following experiment config
    Arguments:
        cfg {yacs.config.CfgNode} -- Loaded experiment config
        logger {logging} -- Simple python logging
    Raises:
        RuntimeError: For wrong config options
    """

    # TODO: predcit with saved model
    # TODO: task train model withoud GA - pre-trained models
    # TODO: pyRadiomics
    # TODO: Segmentation

    datasets = {"TRAIN": {}, "VAL": {}}

    if cfg.MODEL.ARCHITECTURE == "MLP":
        datasets["TRAIN"]["CSV"] = CSVDataset(cfg, logger)
        logger.info(
            f"Train csv dataset loaded! "
            f"{len(datasets['TRAIN']['CSV'])} samples with "
            f"{datasets['TRAIN']['CSV'].n_features} "
            f"features was found!"
        )

        datasets["VAL"]["CSV"] = CSVDataset(cfg, logger, is_validation=True)
        logger.info(
            f"Validation csv dataset loaded! "
            f"{len(datasets['VAL']['CSV'])} samples with "
            f"{datasets['VAL']['CSV'].n_features} features was found!"
        )

    elif cfg.MODEL.ARCHITECTURE == "CNN":
        datasets["TRAIN"]["IMG"] = ImageDataset(cfg, logger)
        logger.info(
            f"Train images dataset loaded! "
            f"{len(datasets['TRAIN']['IMG'])} was found!"
        )

        datasets["VAL"]["IMG"] = ImageDataset(cfg, logger, is_validation=True)
        logger.info(
            f"Validation images dataset loaded! "
            f"{len(datasets['VAL']['IMG'])} was found!"
        )

    elif cfg.MODEL.ARCHITECTURE == "CNN/MLP":
        datasets["TRAIN"]["CSV"] = CSVDataset(cfg, logger)
        logger.info(
            f"Train csv dataset loaded! "
            f"{len(datasets['TRAIN']['CSV'])} samples with "
            f"{datasets['TRAIN']['CSV'].n_features} "
            f"features was found!"
        )
        datasets["TRAIN"]["IMG"] = ImageDataset(cfg, logger)
        logger.info(
            f"Train images dataset loaded! "
            f"{len(datasets['TRAIN']['IMG'])} was found!"
        )

        datasets["VAL"]["CSV"] = CSVDataset(cfg, logger, is_validation=True)
        logger.info(
            f"Validation csv dataset loaded! "
            f"{len(datasets['VAL']['CSV'])} samples with "
            f"{datasets['VAL']['CSV'].n_features} features was found!"
        )
        datasets["VAL"]["IMG"] = ImageDataset(cfg, logger, is_validation=True)
        logger.info(
            f"Validation images dataset loaded! "
            f"{len(datasets['VAL']['IMG'])} was found!"
        )

    else:
        raise RuntimeError(
            "There isn't a valid architecture configured!\n \
                            Check your experiment config"
        )

    ga = GA(cfg, logger, datasets)
    ga.run()


def main():
    """Parse argments and load experiment config"""

    parser = argparse.ArgumentParser(
        description="Neural Network Genetic Algorithm CLI"
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--create-config",
        default="",
        metavar="FILE",
        help="path to store a default config file",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.create_config:
        cfg.freeze()
        export_config(args.create_config)
        print(f"Config Created on {args.create_config}")
        exit(0)

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = os.path.join(output_dir, Path(args.config_file).stem)
    cfg.freeze()

    if output_dir:
        try:
            os.makedirs(cfg.OUTPUT_DIR)
        except FileExistsError:
            pass

    logger = setup_logger("NNGA", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file != "":
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    logger.info(f"CFG: \n{cfg}")

    try:
        train(cfg, logger)
    except Exception:
        msg = f"Failed:\n{traceback.format_exc()}"
        logger.error(msg)


if __name__ == "__main__":
    main()
