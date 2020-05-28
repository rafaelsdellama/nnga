""" Command line inferface for package Neural Network Genetic Algorithm"""
import argparse
import os
from pathlib import Path
import traceback
import webbrowser
from tensorboard import program
from threading import Timer

from nnga.configs import cfg, export_config
from nnga.utils.logger import setup_logger
from nnga import ARCHITECTURES, DATASETS
from nnga.architectures import BACKBONES
from nnga.model_training import ModelTraining
from nnga.genetic_algorithm.ga import GA

# import tensorflow as tf
# Set CPU as available physical device
# tf.config.set_visible_devices([], 'GPU')

TASKS = ["Classification", "Segmentation"]


def train(cfg, logger):
    """
    Create datasets, Models following experiment config
    Arguments:
        cfg {yacs.config.CfgNode} -- Loaded experiment config
        logger {logging} -- Simple python logging
    Raises:
        RuntimeError: For wrong config options
    """

    if cfg.TASK not in TASKS:
        raise RuntimeError(
            "There isn't a valid TASKS!\n \
                            Check your experiment config"
        )

    if cfg.TASK == "Segmentation":
        raise NotImplementedError("Segmentation")

    if cfg.MODEL.ARCHITECTURE not in ARCHITECTURES.keys():
        raise RuntimeError(
            "There isn't a valid architecture configured!\n \
                            Check your experiment config"
        )

    if cfg.MODEL.BACKBONE not in BACKBONES.keys() and cfg.MODEL.BACKBONE \
            not in [
        "MLP",
        "GASearch",
    ]:
        raise RuntimeError(
            "There isn't a valid backbone configured!\n \
                            Check your experiment config"
        )

    # Read datasets
    datasets = {}

    MakeDataset = DATASETS.get(cfg.MODEL.ARCHITECTURE)
    datasets["TRAIN"] = MakeDataset(cfg, logger)
    logger.info(
        f"Train {cfg.MODEL.ARCHITECTURE} dataset loaded! "
        f"{datasets['TRAIN'].n_samples} sample(s) on "
        f"{ len(datasets['TRAIN']) } batch(es) was found!"
    )

    datasets["VAL"] = MakeDataset(cfg, logger, is_validation=True)
    logger.info(
        f"Validation {cfg.MODEL.ARCHITECTURE} dataset loaded! "
        f"{datasets['VAL'].n_samples} sample(s) on "
        f"{ len(datasets['VAL']) } batch(es) was found!"
    )

    if hasattr(datasets["TRAIN"], "scale_parameters"):
        datasets["VAL"].scale_parameters = datasets["TRAIN"].scale_parameters

    if cfg.MODEL.BACKBONE == "GASearch" or cfg.MODEL.FEATURE_SELECTION:
        ga = GA(cfg, logger, datasets)
        ga.run()
    else:
        MakeModel = ARCHITECTURES.get(cfg.MODEL.ARCHITECTURE)
        model = MakeModel(
            cfg,
            logger,
            datasets["TRAIN"].input_shape,
            datasets["TRAIN"].n_classes,
        )
        # Training
        model_trainner = ModelTraining(cfg, model, logger, datasets)

        if cfg.SOLVER.CROSS_VALIDATION:
            cv = model_trainner.cross_validation(save=True)
            logger.info(f"Cross validation statistics:\n{cv}")

        model_trainner.fit()
        model_trainner.compute_metrics(save=True)

    logger.info(f"Model trained!\nCheck the results on {cfg.OUTPUT_DIR}")


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
        launch_tensorboard(cfg.OUTPUT_DIR)
        train(cfg, logger)
    except:
        msg = f"Failed:\n{traceback.format_exc()}"
        logger.error(msg)


def launch_tensorboard(logdir):
    """Open tensorboard."""
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir])
    url = tb.launch()
    print(f"Tensorboard: {url}")
    Timer(1, webbrowser.open_new, args=[url]).start()


if __name__ == "__main__":
    main()
