"""Default experiment config
"""
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.OUTPUT_DIR = "./NNGA_output"  # path to save checkpoints files
_C.TASK = "Classification"

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
_C.GA.CONTINUE_EXEC = False

_C.GA.SEARCH_SPACE = CN()
_C.GA.SEARCH_SPACE.PADDING = ["valid", "same"]
_C.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = 5
_C.GA.SEARCH_SPACE.MAX_CNN_LAYERS = 5
_C.GA.SEARCH_SPACE.UNITS = [20, 50, 80, 100, 150, 200, 250, 500]
_C.GA.SEARCH_SPACE.FILTERS = [4, 5, 6, 7, 8, 9]
_C.GA.SEARCH_SPACE.KERNEL_SIZE = [2, 3, 4, 5, 6]
_C.GA.SEARCH_SPACE.POOL_SIZE = [2, 3, 4, 5]
_C.GA.SEARCH_SPACE.BATCH_SIZE = [3, 4, 5, 6, 7, 8]
_C.GA.SEARCH_SPACE.EPOCHS = [30, 45, 60, 90, 120, 150, 200]
_C.GA.SEARCH_SPACE.DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.4]
_C.GA.SEARCH_SPACE.OPTIMIZER = ["Adam", "SGD"]
_C.GA.SEARCH_SPACE.LEARNING_RATE = [0.0001, 0.001, 0.01, 0.05, 0.1]
_C.GA.SEARCH_SPACE.SCALER = ["Standard", "MinMax"]
_C.GA.SEARCH_SPACE.ACTIVATION_DENSE = ["relu", "tanh"]
_C.GA.SEARCH_SPACE.ACTIVATION_CNN = ["relu", "tanh"]
_C.GA.SEARCH_SPACE.KERNEL_REGULARIZER = [None]
_C.GA.SEARCH_SPACE.KERNEL_INITIALIZER = ["glorot_uniform"]
_C.GA.SEARCH_SPACE.ACTIVITY_REGULARIZER = [None]
_C.GA.SEARCH_SPACE.BIAS_REGULARIZER = [None]

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
_C.DATASET.SCALER = "Standard"

_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "MLP"
_C.MODEL.BACKBONE = "GASearch"
_C.MODEL.FEATURE_SELECTION = False
_C.MODEL.INPUT_SHAPE = (150, 150, 3)
_C.MODEL.DROPOUT = 0.2

_C.SOLVER = CN()
_C.SOLVER.LOSS = "categorical_crossentropy"
_C.SOLVER.METRICS = ["categorical_accuracy"]
_C.SOLVER.CROSS_VALIDATION = True
_C.SOLVER.CROSS_VALIDATION_FOLDS = 10
_C.SOLVER.TEST_SIZE = 0.2
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.EPOCHS = 100
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LEARNING_RATE = 0.0001


def export_config(path):
    """Export config to a yaml file

    Arguments:
        path {str} -- Path file to save yaml
    """
    with open(path, "w") as f:
        f.write(_C.dump())
