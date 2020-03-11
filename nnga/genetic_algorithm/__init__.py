import numpy as np
from nnga.architectures.mlp import MLP
from nnga.architectures.cnn import CNN
from nnga.architectures.cnn_mlp import CNN_MLP

PARAMETERS_INTERVAL = {
    'padding': ['valid', 'same'],
    'units': list(range(20, 101, 5)),
    'filters': list(range(4, 9, 1)),  # filters = 2^i
    'kernel_size': list(range(2, 9, 1)),
    'maxPool': [True, False],
    'pool_size': list(range(2, 6, 1)),
    'batchNormalization': [True, False],
    'batch_size': list(range(3, 8, 1)),  # batch_size = 2^i
    'epochs': list(range(20, 201, 5)),
    'dropout': list(np.arange(0.0, 0.5, 0.1)),
    'optimizer': ['Adam', 'SGD', 'RMSprop', 'Adagrad',
                  'Adadelta', 'Adamax', 'Nadam'],
    'learning_rate': [0.001, 0.0025, 0.005, 0.0075,
                      0.01, 0.025, 0.05, 0.075, 0.1,
                      0.25, 0.5, 0.75, 1.0, 1.1],
    'scaler': ['Standard', 'MinMax'],
    'activation': ['relu', 'tanh'],
    'activate': [True, False],
    'kernel_regularizer': [None, 'l1', 'l2', 'l1_l2'],
    'kernel_initializer': ['Zeros', 'Ones', 'RandomNormal', 'RandomUniform',
                           'TruncatedNormal', 'VarianceScaling', 'Orthogonal',
                           'lecun_uniform', 'glorot_normal', 'glorot_uniform',
                           'he_normal', 'lecun_normal', 'he_uniform'],
    'activity_regularizer': [None, 'l1', 'l2', 'l1_l2'],
    'bias_regularizer': [None, 'l1', 'l2', 'l1_l2'],
    'feature_selection': [True, False]
}

ENCODING = {
    'MLP': {
        'units_0': 'units',
        'dropout_dense_0': 'dropout',

        'activate_1': 'activate',
        'units_1': 'units',
        'dropout_dense_1': 'dropout',

        'activate_2': 'activate',
        'units_2': 'units',
        'dropout_dense_2': 'dropout',

        'activate_3': 'activate',
        'units_3': 'units',
        'dropout_dense_3': 'dropout',

        'activate_4': 'activate',
        'units_4': 'units',
        'dropout_dense_4': 'dropout',

        'activation_dense': 'activation',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'optimizer': 'optimizer',
        'learning_rate': 'learning_rate',
        'scaler': 'scaler',
        'kernel_regularizer': 'kernel_regularizer',
        'kernel_initializer': 'kernel_initializer',
        'activity_regularizer': 'activity_regularizer',
        'bias_regularizer': 'bias_regularizer'
    },

    'CNN': {
        'filters_0': 'filters',
        'kernel_size_0': 'kernel_size',
        'padding_0': 'padding',
        'batchNormalization_0': 'batchNormalization',
        'maxPool_0': 'maxPool',
        'pool_size_0': 'pool_size',
        'dropout_cnn_0': 'dropout',

        'activate_cnn_1': 'activate',
        'filters_1': 'filters',
        'kernel_size_1': 'kernel_size',
        'padding_1': 'padding',
        'batchNormalization_1': 'batchNormalization',
        'maxPool_1': 'maxPool',
        'pool_size_1': 'pool_size',
        'dropout_cnn_1': 'dropout',

        'activate_cnn_2': 'activate',
        'filters_2': 'filters',
        'kernel_size_2': 'kernel_size',
        'padding_2': 'padding',
        'batchNormalization_2': 'batchNormalization',
        'maxPool_2': 'maxPool',
        'pool_size_2': 'pool_size',
        'dropout_cnn_2': 'dropout',

        'activate_cnn_3': 'activate',
        'filters_3': 'filters',
        'kernel_size_3': 'kernel_size',
        'padding_3': 'padding',
        'batchNormalization_3': 'batchNormalization',
        'maxPool_3': 'maxPool',
        'pool_size_3': 'pool_size',
        'dropout_cnn_3': 'dropout',

        'activate_cnn_4': 'activate',
        'filters_4': 'filters',
        'kernel_size_4': 'kernel_size',
        'padding_4': 'padding',
        'batchNormalization_4': 'batchNormalization',
        'maxPool_4': 'maxPool',
        'pool_size_4': 'pool_size',
        'dropout_cnn_4': 'dropout',

        'activate_1': 'activate',
        'units_1': 'units',
        'dropout_dense_1': 'dropout',

        'activate_2': 'activate',
        'units_2': 'units',
        'dropout_dense_2': 'dropout',

        'activate_3': 'activate',
        'units_3': 'units',
        'dropout_dense_3': 'dropout',

        'activate_4': 'activate',
        'units_4': 'units',
        'dropout_dense_4': 'dropout',

        'activation_conv': 'activation',
        'activation_dense': 'activation',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'optimizer': 'optimizer',
        'learning_rate': 'learning_rate',  # 'scaler': 'scaler',
        'kernel_regularizer': 'kernel_regularizer',
        'kernel_initializer': 'kernel_initializer',
        'activity_regularizer': 'activity_regularizer',
        'bias_regularizer': 'bias_regularizer'
    },

    'CNN/MLP': {
        'filters_0': 'filters',
        'kernel_size_0': 'kernel_size',
        'padding_0': 'padding',
        'batchNormalization_0': 'batchNormalization',
        'maxPool_0': 'maxPool',
        'pool_size_0': 'pool_size',
        'dropout_cnn_0': 'dropout',

        'activate_cnn_1': 'activate',
        'filters_1': 'filters',
        'kernel_size_1': 'kernel_size',
        'padding_1': 'padding',
        'batchNormalization_1': 'batchNormalization',
        'maxPool_1': 'maxPool',
        'pool_size_1': 'pool_size',
        'dropout_cnn_1': 'dropout',

        'activate_cnn_2': 'activate',
        'filters_2': 'filters',
        'kernel_size_2': 'kernel_size',
        'padding_2': 'padding',
        'batchNormalization_2':
            'batchNormalization',
        'maxPool_2': 'maxPool',
        'pool_size_2': 'pool_size',
        'dropout_cnn_2': 'dropout',

        'activate_cnn_3': 'activate',
        'filters_3': 'filters',
        'kernel_size_3': 'kernel_size',
        'padding_3': 'padding',
        'batchNormalization_3': 'batchNormalization',
        'maxPool_3': 'maxPool',
        'pool_size_3': 'pool_size',
        'dropout_cnn_3': 'dropout',

        'activate_cnn_4': 'activate',
        'filters_4': 'filters',
        'kernel_size_4': 'kernel_size',
        'padding_4': 'padding',
        'batchNormalization_4': 'batchNormalization',
        'maxPool_4': 'maxPool',
        'pool_size_4': 'pool_size',
        'dropout_cnn_4': 'dropout',

        'activate_1': 'activate',
        'units_1': 'units',
        'dropout_dense_1': 'dropout',
        'activate_2': 'activate',
        'units_2': 'units', 'dropout_dense_2': 'dropout',
        'activate_3': 'activate',
        'units_3': 'units', 'dropout_dense_3': 'dropout',
        'activate_4': 'activate',
        'units_4': 'units',
        'dropout_dense_4': 'dropout',

        'activation_conv': 'activation',
        'activation_dense': 'activation',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'optimizer': 'optimizer',
        'learning_rate': 'learning_rate',
        'scaler': 'scaler',
        'kernel_regularizer': 'kernel_regularizer',
        'kernel_initializer': 'kernel_initializer',
        'activity_regularizer': 'activity_regularizer',
        'bias_regularizer': 'bias_regularizer'
    }
}

MODELS = {
    'MLP': MLP,
    'CNN': CNN,
    'CNN/MLP': CNN_MLP,
}
