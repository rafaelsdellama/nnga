from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
)
from tensorflow.keras.initializers import (
    Zeros,
    Ones,
    RandomNormal,
    RandomUniform,
    TruncatedNormal,
    VarianceScaling,
    Orthogonal,
    lecun_uniform,
    lecun_normal,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
)

from tensorflow.keras.applications import (
    Xception,
    VGG16,
    VGG19,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    NASNetMobile,
    NASNetLarge,
)

from nnga.architectures.segmentation.backbones.unet import unet

REGULARIZERS = {
    "l1": l1(0.01),
    "l2": l2(0.01),
    "l1_l2": l1_l2(l1=0.01, l2=0.01),
}

INICIALIZERS = {
    "Zeros": Zeros(),
    "Ones": Ones(),
    "RandomNormal": RandomNormal(mean=0.0, stddev=0.05),
    "RandomUniform": RandomUniform(minval=-0.05, maxval=0.05),
    "TruncatedNormal": TruncatedNormal(mean=0.0, stddev=0.05),
    "VarianceScaling": VarianceScaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal"
    ),
    "Orthogonal": Orthogonal(gain=1.0),
    "lecun_uniform": lecun_uniform(),
    "lecun_normal": lecun_normal(),
    "glorot_normal": glorot_normal(),
    "glorot_uniform": glorot_uniform(),
    "he_normal": he_normal(),
    "he_uniform": he_uniform(),
}

OPTIMIZERS = {
    "Adam": Adam,
    "SGD": SGD,
    "RMSprop": RMSprop,
    "Adagrad": Adagrad,
    "Adadelta": Adadelta,
    "Adamax": Adamax,
    "Nadam": Nadam,
}

BACKBONES = {
    "Classification": {
        "Xception": Xception,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "ResNet50": ResNet50,
        "ResNet101": ResNet101,
        "ResNet152": ResNet152,
        "ResNet50V2": ResNet50V2,
        "ResNet101V2": ResNet101V2,
        "ResNet152V2": ResNet152V2,
        "InceptionV3": InceptionV3,
        "InceptionResNetV2": InceptionResNetV2,
        "MobileNet": MobileNet,
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "DenseNet201": DenseNet201,
        "NASNetMobile": NASNetMobile,
        "NASNetLarge": NASNetLarge,
    },
    "Segmentation": {
        "unet": unet,
    },
}


def get_backbone(task, backbone):
    """Return a backbone for a specific task and architecture"""
    architecture_by_task = BACKBONES.get(task)
    backbone = architecture_by_task.get(backbone)
    if architecture_by_task is None:
        raise RuntimeError(
            f"There isn't a valid TASKS!\n Check your experiment config\n"
            f"Tasks: {BACKBONES.keys()}"
        )

    elif backbone is None:
        raise RuntimeError(
            f"There isn't a valid backbone configured!\nCheck your "
            f"experiment config\nBackbone for {task}: "
            f"{BACKBONES.get(task).keys()}"
        )
    return backbone
