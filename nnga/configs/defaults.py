"""Default experiment config
"""
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.OUTPUT_DIR = "./NNGA_output"  # path to save checkpoints files

_C.GA = CN()
_C.GA.NRO_MAX_EXEC = 1
_C.GA.NRO_MAX_GEN = 100
_C.GA.POPULATION_SIZE = 100
_C.GA.CROSSOVER_RATE = 0.6
_C.GA.MUTATION_RATE = 0.01
_C.GA.ELITISM = 2
_C.GA.TOURNAMENT_SIZE = 3
_C.GA.HYPERMUTATION = False
_C.GA.HYPERMUTATION_CYCLE_SIZE = 2
_C.GA.HYPERMUTATION_RATE = 3
_C.GA.RANDOM_IMIGRANTS = False
_C.GA.IMIGRATION_RATE = 0.05
_C.GA.SEED = []
_C.GA.FEATURE_SELECTION = False
_C.GA.CONTINUE_EXEC = False

_C.DATASET = CN()

# path to train dataset
_C.DATASET.TRAIN_IMG_PATH = "/PATH/TO/TRAIN/DATASET"
_C.DATASET.TRAIN_CSV_PATH = "/PATH/TO/TRAIN/DATASET/file_name.csv"

# path to validation dataset
_C.DATASET.VAL_IMG_PATH = "/PATH/TO/VAL/DATASET"
_C.DATASET.VAL_CSV_PATH = "/PATH/TO/VAL/DATASET/file_name.csv"

# train augmentation flag
_C.DATASET.TRAIN_AUGMENTATION = True

# flag to suffle train data
_C.DATASET.TRAIN_SHUFFLE = True

# validation augmentation flag
_C.DATASET.VAL_AUGMENTATION = False

# flag to suffle validation data
_C.DATASET.VAL_SHUFFLE = False

_C.DATASET.PRESERVE_IMG_RATIO = True

_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "MLP"
_C.MODEL.INPUT_SHAPE = (200, 200, 3)

_C.SOLVER = CN()
_C.SOLVER.LOSS = 'categorical_crossentropy'
_C.SOLVER.METRICS = ['categorical_accuracy']


def export_config(path):
    """Export config to a yaml file

    Arguments:
        path {str} -- Path file to save yaml
    """
    with open(path, "w") as f:
        f.write(_C.dump())
