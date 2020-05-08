"""Tests for GA."""

import pytest
from pathlib import Path

from nnga.datasets.image_dataset import ImageDataset
from nnga.datasets.csv_dataset import CSVDataset
from nnga.genetic_algorithm.ga import GA
from nnga.utils.logger import setup_logger
from nnga import ROOT
from nnga.configs import cfg

test_directory = Path(ROOT, "tests", "testdata").as_posix()

iris = Path(test_directory, "iris.csv").as_posix()
iris2 = Path(test_directory, "iris2.csv").as_posix()
img_directory = Path(test_directory, "img_dataset").as_posix()
img_features = Path(test_directory, "img_dataset", "features.csv").as_posix()
img_features2 = Path(test_directory, "img_dataset", "features2.csv").as_posix()

pytest_output_directory = "./Pytest_output"

logger = setup_logger("Pytest", pytest_output_directory)


@pytest.mark.parametrize(
    "data_directory, feature_selection, hypermutation, imigrants",
    [(iris, True, False, False),
     (iris, False, False, False),
     (iris2, True, False, False),
     (iris2, False, False, False),
     (iris, True, True, False),
     (iris, False, False, True),
     ],
)
def test_exec_GA_MLP_sucess(data_directory, feature_selection, hypermutation, imigrants):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = 2
    _cfg.GA.POPULATION_SIZE = 4
    _cfg.GA.FEATURE_SELECTION = feature_selection
    _cfg.GA.HYPERMUTATION = hypermutation
    _cfg.GA.RANDOM_IMIGRANTS = imigrants
    _cfg.GA.SEARCH_SPACE.UNITS = [2, 5]
    _cfg.GA.SEARCH_SPACE.EPOCHS = [3, 4]
    _cfg.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = 1

    _cfg.DATASET.TRAIN_CSV_PATH = data_directory
    _cfg.DATASET.VAL_CSV_PATH = data_directory

    _cfg.MODEL.ARCHITECTURE = "MLP"

    datasets = {"TRAIN": {'CSV': CSVDataset(_cfg, logger)},
                "VAL": {'CSV': CSVDataset(_cfg, logger, True)}}

    ga = GA(_cfg, logger, datasets)
    ga.run()


@pytest.mark.parametrize(
    "data_directory",
    [img_directory,
     ],
)
def test_exec_GA_CNN_sucess(data_directory):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = 2
    _cfg.GA.POPULATION_SIZE = 4
    _cfg.GA.SEARCH_SPACE.UNITS = [2, 5]
    _cfg.GA.SEARCH_SPACE.EPOCHS = [3, 4]
    _cfg.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = 1
    _cfg.GA.SEARCH_SPACE.MAX_CNN_LAYERS = 1
    _cfg.GA.SEARCH_SPACE.FILTERS = [1, 2]

    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory

    _cfg.MODEL.ARCHITECTURE = "CNN"

    _cfg.SOLVER.CROSS_VALIDATION_FOLDS = 2
    _cfg.SOLVER.TEST_SIZE = 0.5

    datasets = {"TRAIN": {'IMG': ImageDataset(_cfg, logger)},
                "VAL": {'IMG': ImageDataset(_cfg, logger, True)}}

    ga = GA(_cfg, logger, datasets)
    ga.run()


@pytest.mark.parametrize(
    "data_dir, img_dir, feature_selection",
    [(img_features, img_directory, True),
     (img_features, img_directory, False),
     (img_features2, img_directory, True),
     (img_features2, img_directory, False)
     ],
)
def test_exec_GA_CNN_MLP_sucess(data_dir, img_dir, feature_selection):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = 2
    _cfg.GA.POPULATION_SIZE = 4
    _cfg.GA.FEATURE_SELECTION = feature_selection
    _cfg.GA.SEARCH_SPACE.UNITS = [2, 5]
    _cfg.GA.SEARCH_SPACE.EPOCHS = [3, 4]
    _cfg.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = 1
    _cfg.GA.SEARCH_SPACE.MAX_CNN_LAYERS = 1
    _cfg.GA.SEARCH_SPACE.FILTERS = [1, 2]

    _cfg.DATASET.TRAIN_CSV_PATH = data_dir
    _cfg.DATASET.TRAIN_IMG_PATH = img_dir
    _cfg.DATASET.VAL_CSV_PATH = data_dir
    _cfg.DATASET.VAL_IMG_PATH = img_dir

    _cfg.MODEL.ARCHITECTURE = "CNN/MLP"

    _cfg.SOLVER.CROSS_VALIDATION_FOLDS = 2
    _cfg.SOLVER.TEST_SIZE = 0.5

    datasets = {
        "TRAIN": {
            'CSV': CSVDataset(_cfg, logger),
            'IMG': ImageDataset(_cfg, logger)
        },
        "VAL": {
            'CSV': CSVDataset(_cfg, logger, True),
            'IMG': ImageDataset(_cfg, logger, True)}}

    ga = GA(_cfg, logger, datasets)
    ga.run()


@pytest.mark.parametrize(
    "nro_exec",
    ['wrong', 0, -1
     ],
)
def test_wrong_nro_exec(nro_exec):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_EXEC = nro_exec

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "max_gen",
    ['wrong', 0, -1
     ],
)
def test_wrong_max_gen(max_gen):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = max_gen

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pop_size",
    ['wrong', 0, -1
     ],
)
def test_wrong_pop_size(pop_size):
    _cfg = cfg.clone()
    _cfg.GA.POPULATION_SIZE = pop_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "crossover_rate",
    ['wrong', 2, 1, 0, -1
     ],
)
def test_wrong_crossover_rate(crossover_rate):
    _cfg = cfg.clone()
    _cfg.GA.CROSSOVER_RATE = crossover_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "mutation_rate",
    ['wrong', 2, 1, 0, -1
     ],
)
def test_wrong_mutation_rate(mutation_rate):
    _cfg = cfg.clone()
    _cfg.GA.MUTATION_RATE = mutation_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pop_size, elitism",
    [(5, 'wrong'),
     (5, 0.1),
     (5, -1),
     (5, 10),
     ],
)
def test_wrong_elitism(pop_size, elitism):
    _cfg = cfg.clone()
    _cfg.GA.POPULATION_SIZE = pop_size
    _cfg.GA.ELITISM = elitism

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pop_size, tournament_size",
    [(5, 'wrong'),
     (5, 0.1),
     (5, -1),
     (5, 10),
     ],
)
def test_wrong_tournament_size(pop_size, tournament_size):
    _cfg = cfg.clone()
    _cfg.GA.POPULATION_SIZE = pop_size
    _cfg.GA.TOURNAMENT_SIZE = tournament_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "hypermutation",
    ['wrong', 0.1, 1,
     ],
)
def test_wrong_hypermutation(hypermutation):
    _cfg = cfg.clone()
    _cfg.GA.HYPERMUTATION = hypermutation

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "max_gen, cycle_size",
    [(5, 'wrong'),
     (5, 0.1),
     (5, -1),
     (5, 0),
     (5, 10),
     ],
)
def test_wrong_cycle_size(max_gen, cycle_size):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = max_gen
    _cfg.GA.HYPERMUTATION_CYCLE_SIZE = cycle_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "hypermutation_rate",
    ['wrong', 0, -1,
     ],
)
def test_wrong_hypermutation_rate(hypermutation_rate):
    _cfg = cfg.clone()
    _cfg.GA.HYPERMUTATION_RATE = hypermutation_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "rd_imigrantes",
    ['wrong', 0.1, 1,
     ],
)
def test_wrong_rd_imigrantes(rd_imigrantes):
    _cfg = cfg.clone()
    _cfg.GA.RANDOM_IMIGRANTS = rd_imigrantes

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "imigration_rate",
    ['wrong', 0, 1, 1.1,
     ],
)
def test_wrong_imigration_rate(imigration_rate):
    _cfg = cfg.clone()
    _cfg.GA.IMIGRATION_RATE = imigration_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "seeds, nro_exec",
    [('wrong', 5),
     ([1, 1.1], 2),
     ([1.1], 1),
     ([1, 2], 5)
     ],
)
def test_wrong_seed(seeds, nro_exec):
    _cfg = cfg.clone()
    _cfg.GA.SEED = seeds
    _cfg.GA.NRO_MAX_EXEC = nro_exec

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "feature_selection",
    ['wrong', 0.1, 1,
     ],
)
def test_wrong_feature_selection(feature_selection):
    _cfg = cfg.clone()
    _cfg.GA.FEATURE_SELECTION = feature_selection

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "continue_exec",
    ['wrong', 0.1, 1,
     ],
)
def test_wrong_continue_exec(continue_exec):
    _cfg = cfg.clone()
    _cfg.GA.CONTINUE_EXEC = continue_exec

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "padding",
    ['wrong', 'valid', "",
     ['wrong', 'valid'],
     ['valid', 'wrong'],
     ],
)
def test_wrong_padding(padding):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.PADDING = padding

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "dense_layers",
    ['wrong', -1, 100
     ],
)
def test_wrong_dense_layers(dense_layers):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = dense_layers

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "cnn_layers",
    ['wrong', -1, 100
     ],
)
def test_wrong_dense_layers(cnn_layers):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.MAX_CNN_LAYERS = cnn_layers

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "units",
    ['wrong', 1,
     [0, 10],
     [10, 'wrong'],
     [10, 10000000],
     []
     ],
)
def test_wrong_units(units):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.UNITS = units

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "filters",
    ['wrong', 1,
     [0, 10],
     [10, 'wrong'],
     [10, 10000000],
     []
     ],
)
def test_wrong_filters(filters):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.FILTERS = filters

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "kernel_size",
    ['wrong', 1,
     [0, 10],
     [10, 'wrong'],
     [10, 10000000],
     []
     ],
)
def test_wrong_kernel_size(kernel_size):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.KERNEL_SIZE = kernel_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pool_size",
    ['wrong', 1,
     [0, 10],
     [10, 'wrong'],
     [10, 10000000],
     []
     ],
)
def test_wrong_pool_size(pool_size):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.POOL_SIZE = pool_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "batch_size",
    ['wrong', 1,
     [0, 10],
     [10, 'wrong'],
     [10, 10000000],
     []
     ],
)
def test_wrong_batch_size(batch_size):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.BATCH_SIZE = batch_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "epochs",
    ['wrong', 1,
     [0, 10],
     [10, 'wrong'],
     [10, 10000000],
     []
     ],
)
def test_wrong_epochs(epochs):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.EPOCHS = epochs

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "dropout",
    ['wrong', 1,
     [0, 1.1],
     [1, 'wrong'],
     [-1, 1],
     []
     ],
)
def test_wrong_dropout(dropout):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.DROPOUT = dropout

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "opt",
    ['wrong', 'Adam',
     ['wrong', 'Adam'],
     ['Adam', 'wrong'],
     ],
)
def test_wrong_opt(opt):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.OPTIMIZER = opt

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "lr",
    ['wrong', 1,
     [0, 1],
     [1, 'wrong'],
     [-1, 1],
     [0.1, 1.1],
     []
     ],
)
def test_wrong_lr(lr):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.LEARNING_RATE = lr

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "scaler",
    ['wrong', 'Standard',
     ['wrong', 'Standard'],
     ['Standard', 'wrong'],
     ],
)
def test_wrong_scaler(scaler):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.SCALER = scaler

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "activate",
    ['wrong', 'relu',
     ['wrong', 'relu'],
     ['relu', 'wrong'],
     ],
)
def test_wrong_activate_dense(activate):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.ACTIVATION_DENSE = activate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "activate",
    ['wrong', 'relu',
     ['wrong', 'relu'],
     ['relu', 'wrong'],
     ],
)
def test_wrong_activate_cnn(activate):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.ACTIVATION_CNN = activate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "kernel_reg",
    ['wrong', 'l1',
     ['wrong', 'l1'],
     ['l1', 'wrong'],
     ],
)
def test_wrong_kernel_reg(kernel_reg):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.KERNEL_REGULARIZER = kernel_reg

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "kernel_init",
    ['wrong', 'glorot_uniform',
     ['wrong', 'glorot_uniform'],
     ['glorot_uniform', 'wrong'],
     ],
)
def test_wrong_kernel_init(kernel_init):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.KERNEL_INITIALIZER = kernel_init

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "activity_reg",
    ['wrong', 'l1',
     ['wrong', 'l1'],
     ['l1', 'wrong'],
     ],
)
def test_wrong_activity_reg(activity_reg):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.ACTIVITY_REGULARIZER = activity_reg

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "bias_reg",
    ['wrong', 'l1',
     ['wrong', 'l1'],
     ['l1', 'wrong'],
     ],
)
def test_wrong_bias_reg(bias_reg):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.BIAS_REGULARIZER = bias_reg

    with pytest.raises(ValueError):
        GA(_cfg, None, None)
