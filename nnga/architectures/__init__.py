from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, \
    Adadelta, Adamax, Nadam
from tensorflow.keras.initializers import Zeros, Ones, RandomNormal, \
    RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal, \
    lecun_uniform, lecun_normal, glorot_normal, glorot_uniform, he_normal, \
    he_uniform


REGULARIZER = {
    'l1': l1(0.01),
    'l2': l2(0.01),
    'l1_l2': l1_l2(l1=0.01, l2=0.01),
}

INICIALIZER = {
    'Zeros': Zeros(),
    'Ones': Ones(),
    'RandomNormal': RandomNormal(mean=0.0, stddev=0.05),
    'RandomUniform': RandomUniform(minval=-0.05, maxval=0.05),
    'TruncatedNormal': TruncatedNormal(mean=0.0, stddev=0.05),
    'VarianceScaling': VarianceScaling(scale=1.0, mode='fan_in',
                                       distribution='truncated_normal'),
    'Orthogonal': Orthogonal(gain=1.0),
    'lecun_uniform': lecun_uniform(),
    'lecun_normal': lecun_normal(),
    'glorot_normal': glorot_normal(),
    'glorot_uniform': glorot_uniform(),
    'he_normal': he_normal(),
    'he_uniform': he_uniform(),

}

OPTIMIZER = {
    'Adam': Adam,
    'SGD': SGD,
    'RMSprop': RMSprop,
    'Adagrad': Adagrad,
    'Adadelta': Adadelta,
    'Adamax': Adamax,
    'Nadam': Nadam,
}


def create_optimizer(op, lr):
    return OPTIMIZER.get(op)(lr=lr)
