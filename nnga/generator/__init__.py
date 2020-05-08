from nnga.generator.mlp import MLP_Generator
from nnga.generator.cnn import CNN_Generator
from nnga.generator.cnn_mlp import CNN_MLP_Generator


GENERATOR = {
    "MLP": MLP_Generator,
    "CNN": CNN_Generator,
    "CNN/MLP": CNN_MLP_Generator,
}
