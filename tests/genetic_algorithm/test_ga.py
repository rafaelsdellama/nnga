"""Tests for GA."""
import pytest
from pathlib import Path
from nnga import ROOT
from nnga.configs import cfg
from nnga.genetic_algorithm.ga import GA
from nnga import get_dataset
from nnga.utils.logger import setup_logger

test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_mnist = Path(test_directory, "datasets", "classification",
                 "mnist").as_posix()
features_mnist = Path(
    test_directory, "datasets", "classification", "mnist", "features.csv"
).as_posix()

pytest_output_directory = "./Pytest_output"
logger = setup_logger("Pytest", pytest_output_directory)


@pytest.mark.parametrize(
    "architecture, backbone, feature_selection, hypermutation, imigrants",
    [
        ("MLP", "MLP", True, False, False),
        ("MLP", "MLP", True, True, False),
        ("MLP", "MLP", True, False, True),
        ("MLP", "GASearch", False, False, False),
        ("MLP", "GASearch", False, True, False),
        ("MLP", "GASearch", False, False, True),
        ("MLP", "GASearch", True, False, False),
        ("MLP", "GASearch", True, True, False),
        ("MLP", "GASearch", True, False, True),
        ("CNN", "GASearch", False, False, False),
        ("CNN", "GASearch", False, True, False),
        ("CNN", "GASearch", False, False, True),
        ("CNN/MLP", "VGG16", True, False, False),
        ("CNN/MLP", "VGG16", True, True, False),
        ("CNN/MLP", "VGG16", True, False, True),
        ("CNN/MLP", "GASearch", True, False, False),
        ("CNN/MLP", "GASearch", True, True, False),
        ("CNN/MLP", "GASearch", True, False, True),
        ("CNN/MLP", "GASearch", False, False, False),
        ("CNN/MLP", "GASearch", False, True, False),
        ("CNN/MLP", "GASearch", False, False, True),
    ],
)
def test_fit_model_GA(
    architecture, backbone, feature_selection, hypermutation, imigrants
):
    _cfg = cfg.clone()
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.TASK = "Classification"
    _cfg.DATASET.TRAIN_IMG_PATH = img_mnist
    _cfg.DATASET.TRAIN_CSV_PATH = features_mnist
    _cfg.DATASET.VAL_IMG_PATH = img_mnist
    _cfg.DATASET.VAL_CSV_PATH = features_mnist
    _cfg.MODEL.INPUT_SHAPE = (64, 64, 3)
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = feature_selection
    _cfg.GA.NRO_MAX_GEN = 4
    _cfg.GA.POPULATION_SIZE = 3
    _cfg.GA.HYPERMUTATION = hypermutation
    _cfg.GA.HYPERMUTATION_CYCLE_SIZE = 1
    _cfg.GA.RANDOM_IMIGRANTS = imigrants
    _cfg.GA.IMIGRATION_RATE = 0.5
    _cfg.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = 1
    _cfg.GA.SEARCH_SPACE.MAX_CNN_LAYERS = 1
    _cfg.GA.SEARCH_SPACE.UNITS = [3, 4]
    _cfg.GA.SEARCH_SPACE.EPOCHS = [1, 2]

    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    datasets = {
        "TRAIN": MakeDataset(_cfg, logger),
        "VAL": MakeDataset(_cfg, logger, is_validation=True),
    }

    if hasattr(datasets["TRAIN"], "scale_parameters"):
        datasets["VAL"].scale_parameters = datasets["TRAIN"].scale_parameters

    ga = GA(_cfg, logger, datasets)
    ga.run()


@pytest.mark.parametrize(
    "max_gen", ["wrong", 0, -1],
)
def test_wrong_max_gen(max_gen):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = max_gen

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pop_size", ["wrong", 0, -1],
)
def test_wrong_pop_size(pop_size):
    _cfg = cfg.clone()
    _cfg.GA.POPULATION_SIZE = pop_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "crossover_rate", ["wrong", 2, 1, 0, -1],
)
def test_wrong_crossover_rate(crossover_rate):
    _cfg = cfg.clone()
    _cfg.GA.CROSSOVER_RATE = crossover_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "mutation_rate", ["wrong", 2, 1, 0, -1],
)
def test_wrong_mutation_rate(mutation_rate):
    _cfg = cfg.clone()
    _cfg.GA.MUTATION_RATE = mutation_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pop_size, elitism", [(5, "wrong"), (5, 0.1), (5, -1), (5, 10),],
)
def test_wrong_elitism(pop_size, elitism):
    _cfg = cfg.clone()
    _cfg.GA.POPULATION_SIZE = pop_size
    _cfg.GA.ELITISM = elitism

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pop_size, tournament_size", [(5, "wrong"), (5, 0.1), (5, -1), (5, 10),],
)
def test_wrong_tournament_size(pop_size, tournament_size):
    _cfg = cfg.clone()
    _cfg.GA.POPULATION_SIZE = pop_size
    _cfg.GA.TOURNAMENT_SIZE = tournament_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "hypermutation", ["wrong", 0.1, 1,],
)
def test_wrong_hypermutation(hypermutation):
    _cfg = cfg.clone()
    _cfg.GA.HYPERMUTATION = hypermutation

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "max_gen, cycle_size", [(5, "wrong"), (5, 0.1), (5, -1), (5, 0), (5, 10),],
)
def test_wrong_cycle_size(max_gen, cycle_size):
    _cfg = cfg.clone()
    _cfg.GA.NRO_MAX_GEN = max_gen
    _cfg.GA.HYPERMUTATION_CYCLE_SIZE = cycle_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "hypermutation_rate", ["wrong", 0, -1,],
)
def test_wrong_hypermutation_rate(hypermutation_rate):
    _cfg = cfg.clone()
    _cfg.GA.HYPERMUTATION_RATE = hypermutation_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "rd_imigrantes", ["wrong", 0.1, 1,],
)
def test_wrong_rd_imigrantes(rd_imigrantes):
    _cfg = cfg.clone()
    _cfg.GA.RANDOM_IMIGRANTS = rd_imigrantes

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "imigration_rate", ["wrong", 0, 1, 1.1,],
)
def test_wrong_imigration_rate(imigration_rate):
    _cfg = cfg.clone()
    _cfg.GA.IMIGRATION_RATE = imigration_rate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "seed", [("wrong"), ([1, 1.1]), ([1.1]), ([1, 2])],
)
def test_wrong_seed(seed):
    _cfg = cfg.clone()
    _cfg.GA.SEED = seed

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "continue_exec", ["wrong", 0.1, 1,],
)
def test_wrong_continue_exec(continue_exec):
    _cfg = cfg.clone()
    _cfg.GA.CONTINUE_EXEC = continue_exec

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "padding", ["wrong", "valid", "", ["wrong", "valid"], ["valid", "wrong"],],
)
def test_wrong_padding(padding):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.PADDING = padding

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "dense_layers", ["wrong", -1, 100],
)
def test_wrong_dense_layers(dense_layers):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.MAX_DENSE_LAYERS = dense_layers

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "cnn_layers", ["wrong", -1, 100],
)
def test_wrong_cnn_layers(cnn_layers):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.MAX_CNN_LAYERS = cnn_layers

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "units", ["wrong", 1, [0, 10], [10, "wrong"], [10, 10000000], []],
)
def test_wrong_units(units):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.UNITS = units

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "filters", ["wrong", 1, [0, 10], [10, "wrong"], [10, 10000000], []],
)
def test_wrong_filters(filters):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.FILTERS = filters

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "kernel_size", ["wrong", 1, [0, 10], [10, "wrong"], [10, 10000000], []],
)
def test_wrong_kernel_size(kernel_size):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.KERNEL_SIZE = kernel_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "pool_size", ["wrong", 1, [0, 10], [10, "wrong"], [10, 10000000], []],
)
def test_wrong_pool_size(pool_size):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.POOL_SIZE = pool_size

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "epochs", ["wrong", 1, [0, 10], [10, "wrong"], [10, 10000000], []],
)
def test_wrong_epochs(epochs):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.EPOCHS = epochs

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "dropout", ["wrong", 1, [0, 1.1], [1, "wrong"], [-1, 1], []],
)
def test_wrong_dropout(dropout):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.DROPOUT = dropout

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "opt", ["wrong", "Adam", ["wrong", "Adam"], ["Adam", "wrong"],],
)
def test_wrong_opt(opt):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.OPTIMIZER = opt

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "lr", ["wrong", 1, [0, 1], [1, "wrong"], [-1, 1], [0.1, 1.1], []],
)
def test_wrong_lr(lr):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.LEARNING_RATE = lr

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "scaler",
    ["wrong", "Standard", ["wrong", "Standard"], ["Standard", "wrong"],],
)
def test_wrong_scaler(scaler):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.SCALER = scaler

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "activate", ["wrong", "relu", ["wrong", "relu"], ["relu", "wrong"],],
)
def test_wrong_activate_dense(activate):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.ACTIVATION_DENSE = activate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "activate", ["wrong", "relu", ["wrong", "relu"], ["relu", "wrong"],],
)
def test_wrong_activate_cnn(activate):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.ACTIVATION_CNN = activate

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "kernel_reg", ["wrong", "l1", ["wrong", "l1"], ["l1", "wrong"],],
)
def test_wrong_kernel_reg(kernel_reg):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.KERNEL_REGULARIZER = kernel_reg

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "kernel_init",
    [
        "wrong",
        "glorot_uniform",
        ["wrong", "glorot_uniform"],
        ["glorot_uniform", "wrong"],
    ],
)
def test_wrong_kernel_init(kernel_init):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.KERNEL_INITIALIZER = kernel_init

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "activity_reg", ["wrong", "l1", ["wrong", "l1"], ["l1", "wrong"],],
)
def test_wrong_activity_reg(activity_reg):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.ACTIVITY_REGULARIZER = activity_reg

    with pytest.raises(ValueError):
        GA(_cfg, None, None)


@pytest.mark.parametrize(
    "bias_reg", ["wrong", "l1", ["wrong", "l1"], ["l1", "wrong"],],
)
def test_wrong_bias_reg(bias_reg):
    _cfg = cfg.clone()
    _cfg.GA.SEARCH_SPACE.BIAS_REGULARIZER = bias_reg

    with pytest.raises(ValueError):
        GA(_cfg, None, None)
