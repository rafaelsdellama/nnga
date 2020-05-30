CHECK_PARAMETERS_INTERVAL = {
    "PADDING": ["valid", "same"],
    "UNITS": {"MIN": 1, "MAX": 5000},
    "FILTERS": {"MIN": 1, "MAX": 10},
    "KERNEL_SIZE": {"MIN": 1, "MAX": 20},
    "MAX_POOL": [True, False],
    "POOL_SIZE": {"MIN": 2, "MAX": 20},
    "BATCH_NORMALIZATION": [True, False],
    "EPOCHS": {"MIN": 1, "MAX": 2000},
    "DROPOUT": {"MIN": 0.0, "MAX": 1.0},
    "OPTIMIZER": [
        "Adam",
        "SGD",
        "RMSprop",
        "Adagrad",
        "Adadelta",
        "Adamax",
        "Nadam",
    ],
    "LEARNING_RATE": {"MIN": 1e-5, "MAX": 1},
    "SCALER": ["Standard", "MinMax"],
    "ACTIVATION_DENSE": [
        "relu",
        "tanh",
        "elu",
        "softmax",
        "selu",
        "softplus",
        "softsign",
        "sigmoid",
        "hard_sigmoid",
        "exponential",
        "linear",
    ],
    "ACTIVATION_CNN": [
        "relu",
        "tanh",
        "elu",
        "softmax",
        "selu",
        "softplus",
        "softsign",
        "sigmoid",
        "hard_sigmoid",
        "exponential",
        "linear",
    ],
    "ACTIVATE": [True, False],
    "KERNEL_REGULARIZER": [None, "l1", "l2", "l1_l2"],
    "KERNEL_INITIALIZER": [
        "Zeros",
        "Ones",
        "RandomNormal",
        "RandomUniform",
        "TruncatedNormal",
        "VarianceScaling",
        "Orthogonal",
        "lecun_uniform",
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "lecun_normal",
        "he_uniform",
    ],
    "ACTIVITY_REGULARIZER": [None, "l1", "l2", "l1_l2"],
    "BIAS_REGULARIZER": [None, "l1", "l2", "l1_l2"],
    "MAX_DENSE_LAYERS": {"MIN": 0, "MAX": 50},
    "MAX_CNN_LAYERS": {"MIN": 0, "MAX": 50},
}

PARAMETERS_INTERVAL = {
    "PADDING": {"type": "string", "value": None},
    "UNITS": {"type": "numbers", "value": None},
    "FILTERS": {"type": "numbers", "value": None},
    "KERNEL_SIZE": {"type": "numbers", "value": None},
    "MAX_POOL": {"type": "string", "value": [True, False]},
    "POOL_SIZE": {"type": "numbers", "value": None},
    "BATCH_NORMALIZATION": {"type": "string", "value": [True, False]},
    "EPOCHS": {"type": "numbers", "value": None},
    "DROPOUT": {"type": "numbers", "value": None},
    "OPTIMIZER": {"type": "string", "value": None},
    "LEARNING_RATE": {"type": "numbers", "value": None},
    "SCALER": {"type": "string", "value": None},
    "ACTIVATION_DENSE": {"type": "string", "value": None},
    "ACTIVATION_CNN": {"type": "string", "value": None},
    "ACTIVATE": {"type": "string", "value": [True, False]},
    "KERNEL_REGULARIZER": {"type": "string", "value": None},
    "KERNEL_INITIALIZER": {"type": "string", "value": None},
    "ACTIVITY_REGULARIZER": {"type": "string", "value": None},
    "BIAS_REGULARIZER": {"type": "string", "value": None},
    "FEATURE_SELECTION": {"type": "string", "value": [True, False]},
    "MAX_DENSE_LAYERS": {"type": "number", "value": None},
    "MAX_CNN_LAYERS": {"type": "number", "value": None},
}

PARAMETERS_ARCHITECTURE = {
    "DEFAULT": {
        "activation_dense": "ACTIVATION_DENSE",
        "epochs": "EPOCHS",
        "optimizer": "OPTIMIZER",
        "learning_rate": "LEARNING_RATE",
        "kernel_regularizer": "KERNEL_REGULARIZER",
        "kernel_initializer": "KERNEL_INITIALIZER",
        "activity_regularizer": "ACTIVITY_REGULARIZER",
        "bias_regularizer": "BIAS_REGULARIZER",
    },
    "LAYER": {
        "DENSE_LAYER": {
            "activate_dense": "ACTIVATE",
            "units": "UNITS",
            "dropout_dense": "DROPOUT",
        },
        "CNN_LAYER": {
            "activate_cnn": "ACTIVATE",
            "filters": "FILTERS",
            "kernel_size": "KERNEL_SIZE",
            "padding": "PADDING",
            "batch_normalization": "BATCH_NORMALIZATION",
            "max_pool": "MAX_POOL",
            "pool_size": "POOL_SIZE",
            "dropout_cnn": "DROPOUT",
        },
    },
    "MLP": {"scaler": "SCALER"},
    "CNN": {"activation_cnn": "ACTIVATION_CNN"},
}


def set_parameters(parameters, architecture, backbone, features):
    """Set the network parameters to be search by the GA
    Parameters
        ----------
            parameters: Dict
                Parameters values provided by cfg
            architecture: str
                Architecture configured in cfg
            backbone: str
                Backbone configured in cfg
            features: list
                List with all features to be selected by GA
        Returns
        -------
        ENCODING: dict with parameters key to create the indivs
        NAME_FEATURES_SELECTION: dict to map the feature to feature name
        PARAMETERS_INTERVAL: dict with the values for each parameter
    """
    for key, value in parameters.items():
        PARAMETERS_INTERVAL[key]["value"] = value

    __check_parameters()
    ENCODING, NAME_FEATURES_SELECTION = create_encoding(
        architecture, backbone, features
    )

    return ENCODING, NAME_FEATURES_SELECTION, PARAMETERS_INTERVAL


def __check_parameters():
    """Check the parameters provided by cfg"""
    for key, parameter in PARAMETERS_INTERVAL.items():
        if key not in CHECK_PARAMETERS_INTERVAL.keys():
            continue

        if parameter["type"] == "string":
            if len(parameter["value"]) == 0:
                raise ValueError(
                    f"GA.SEARCH_SPACE.{key} parameters cannot "
                    f"be empty."
                    f"Check the documentation! \n"
                    f"Options: {CHECK_PARAMETERS_INTERVAL[key]}"
                )

            if not all(
                item in CHECK_PARAMETERS_INTERVAL[key]
                for item in parameter["value"]
            ):
                raise ValueError(
                    f"GA.SEARCH_SPACE.{key} parameters with "
                    f"invalid value. "
                    f"Check the documentation! \n"
                    f"Options: {CHECK_PARAMETERS_INTERVAL[key]}"
                )

        elif parameter["type"] == "numbers":
            if (
                type(parameter["value"]) != list
                or not all(
                    type(i) == int or type(i) == float
                    for i in parameter["value"]
                )
                or min(parameter["value"])
                < CHECK_PARAMETERS_INTERVAL[key]["MIN"]
                or max(parameter["value"])
                > CHECK_PARAMETERS_INTERVAL[key]["MAX"]
            ):
                raise ValueError(
                    f"GA.SEARCH_SPACE.{key} parameters with "
                    f"invalid value. "
                    f"Check the documentation! \n"
                    f"Options: {CHECK_PARAMETERS_INTERVAL[key]}"
                )

        elif parameter["type"] == "number":
            if (
                (
                    type(parameter["value"]) != float
                    and type(parameter["value"]) != int
                )
                or parameter["value"] < CHECK_PARAMETERS_INTERVAL[key]["MIN"]
                or parameter["value"] > CHECK_PARAMETERS_INTERVAL[key]["MAX"]
            ):
                raise ValueError(
                    f"GA.SEARCH_SPACE.{key} parameters with "
                    f"invalid value. "
                    f"Check the documentation! \n"
                    f"{CHECK_PARAMETERS_INTERVAL[key]}"
                )


def create_encoding(architecture, backbone, features):
    """Create encoding dict to be used by the GA for generate the indivs
    Parameters
        ----------
            architecture: str
                Architecture configured in cfg
            backbone: str
                Backbone configured in cfg
            features: list
                List with all features to be selected by GA
        Returns
        -------
    """
    ENCODING = {}
    NAME_FEATURES_SELECTION = {}
    if backbone == "GASearch":
        ENCODING.update(PARAMETERS_ARCHITECTURE["DEFAULT"])

        if architecture == "CNN" or architecture == "CNN/MLP":
            ENCODING.update(PARAMETERS_ARCHITECTURE["CNN"])

            # Create cnn layers
            for i in range(PARAMETERS_INTERVAL["MAX_CNN_LAYERS"]["value"]):
                for key, value in PARAMETERS_ARCHITECTURE["LAYER"][
                    "CNN_LAYER"
                ].items():
                    ENCODING[key + "_" + str(i)] = value
            ENCODING.pop("activate_cnn_0")

        # Create dense layers
        for i in range(PARAMETERS_INTERVAL["MAX_DENSE_LAYERS"]["value"]):
            for key, value in PARAMETERS_ARCHITECTURE["LAYER"][
                "DENSE_LAYER"
            ].items():
                ENCODING[key + "_" + str(i)] = value
        ENCODING.pop("activate_dense_0")

        if architecture == "MLP" or architecture == "CNN/MLP":
            ENCODING.update(PARAMETERS_ARCHITECTURE["MLP"])

    # Features selection
    for i, name in enumerate(features):
        ENCODING[f"feature_selection_{i}"] = "FEATURE_SELECTION"
        NAME_FEATURES_SELECTION[f"feature_selection_{i}"] = name

    return ENCODING, NAME_FEATURES_SELECTION
