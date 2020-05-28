# Getting Started: The Basics

## Using NNGA **C**ommand **Li**ne **I**nterface

Once time NNGA was installed we are abble to use the **nnga-cli** application to execute our experiments, something like that:

```console
nnga-cli --config-file experiments/baseline.yaml
```

If everything goes well with your experiment the training process will start.

```console
2020-03-10 17:22:59,359 NNGA INFO: Namespace(config_file='experiments/baseline.yaml', create_config='', opts=[])
2020-03-10 17:22:59,359 NNGA INFO: Loaded configuration file experiments/baseline.yaml
2020-03-10 17:22:59,360 NNGA INFO:
...
2020-03-10 17:29:05,879 NNGA INFO: Generation: 1
Fitness best indiv: 0.6121212121212122
Best indivs: [2, 0]
Fitness mean: 0.5142742643829601
Mutation rate: 0.01
```

### Create an experiment
To create an experiment could simple make a copy from an existent file or use **nnga-cli** to create a initial experiment file, with the following command.

```console
nnga-cli --create-config experiments/first.yaml
```

Here we could see an example of experiment file:
```yaml
TASK: "Classification"

VERBOSE: 0

# Dataset experiment configuration
DATASET:
  SCALER: "Standard"
  PRESERVE_IMG_RATIO: true
  TRAIN_AUGMENTATION: true
  TRAIN_CSV_PATH: /PATH/TO/TRAIN/DATASET/file_name.csv
  TRAIN_IMG_PATH: /PATH/TO/TRAIN/DATASET
  TRAIN_SHUFFLE: true
  VAL_AUGMENTATION: false
  VAL_CSV_PATH: /PATH/TO/VAL/DATASET/file_name.csv
  VAL_IMG_PATH: /PATH/TO/VAL/DATASET
  VAL_SHUFFLE: false

# Genetic Algorithm configuration
GA:
  CONTINUE_EXEC: false
  SEED: 0
  NRO_MAX_GEN: 100  
  POPULATION_SIZE: 100
  CROSSOVER_RATE: 0.6
  MUTATION_RATE: 0.01
  ELITISM: 2
  HYPERMUTATION: false
  HYPERMUTATION_CYCLE_SIZE: 2
  HYPERMUTATION_RATE: 3
  RANDOM_IMIGRANTS: false
  IMIGRATION_RATE: 0.05  
  TOURNAMENT_SIZE: 3  
  SEARCH_SPACE:
    PADDING: ['valid', 'same']
    MAX_DENSE_LAYERS: 5
    MAX_CNN_LAYERS: 5
    UNITS: [20, 50, 80, 100, 150, 200, 250, 500]
    FILTERS: [4, 5, 6, 7, 8, 9]
    KERNEL_SIZE: [2, 3, 4, 5, 6]
    POOL_SIZE: [2, 3, 4, 5]
    BATCH_SIZE: [3, 4, 5, 6, 7, 8]
    EPOCHS: [30, 45, 60, 90, 120, 150, 200]
    DROPOUT: [0.0, 0.1, 0.2, 0.3, 0.4]
    OPTIMIZER: ['Adam', 'SGD']
    LEARNING_RATE: [0.0001, 0.001, 0.01, 0.05, 0.1]
    SCALER: ['Standard', 'MinMax']
    ACTIVATION_DENSE: ['relu', 'tanh']
    ACTIVATION_CNN: ['relu', 'tanh']
    KERNEL_REGULARIZER: [None]
    KERNEL_INITIALIZER: ['glorot_uniform']
    ACTIVITY_REGULARIZER: [None]
    BIAS_REGULARIZER: [None]

# Model experiment configuration
MODEL:
  ARCHITECTURE: "MLP"
  BACKBONE: "GASearch"
  FEATURE_SELECTION: false  
  INPUT_SHAPE: [150, 150, 3]
  DROPOUT: 0.2

# Solver experiment configuration
SOLVER:
  LOSS: "categorical_crossentropy"
  METRICS: ["categorical_accuracy"]
  CROSS_VALIDATION: true
  CROSS_VALIDATION_FOLDS: 10
  TEST_SIZE: 0.2
  BATCH_SIZE: 128
  EPOCHS: 100
  OPTIMIZER: "Adam"
  BASE_LEARNING_RATE: 0.0001

OUTPUT_DIR: ./NNGA_output
```

#### A tour to all experiments options

Let's go to take a closer look of all options available in an experiment.  

The first important option is about what task the experiment will do. Nowadays 2 options are available Classification and Segmentation.
```yaml
# Task experiment configuration
# Options available here are:
# - Classification
# - Segmentation
TASK: "Classification"
```
The second section is about messages:
```yaml
# 0: quiet, 
# 1: show callback messages
VERBOSE: 0
```
The next section is about datasets: 
```yaml
# Dataset experiment configuration
DATASET:
  #flag to preserve the aspect ratio when the image is resized to INPUT_SHAPE
  #If true, resize the image while maintaining the aspect ratio, adding a black background
  #If false, resize the image to INPUT_SHAPE without worrying about maintaining the aspect ratio
  PRESERVE_IMG_RATIO: true
  
  # Standardization method used to scale the csv data
  # Is used only when the BACKBONE is not "GASearch" and 
  # ARCHITECTURE are MLP or CNN/MLP
  SCALER: 'Standard'

  # path to train dataset
  # you could use an env var to root of ml-datasets repo, like:
  # TRAIN: "${PATH_TO_ML_DATASET_REPO}/PATH/TO/TRAIN/DATASET"
  # or use absolute paths
  TRAIN_CSV_PATH: /PATH/TO/TRAIN/DATASET/file_name.csv
  TRAIN_IMG_PATH: /PATH/TO/TRAIN/DATASET
  
  # flag to make data augmentation on train images
  TRAIN_AUGMENTATION: true
  
  # flag to shuffle train images every epoch
  TRAIN_SHUFFLE: true

  # path to validation dataset
  # you could use an env var to root of ml-datasets repo, like:
  # TRAIN: "${PATH_TO_ML_DATASET_REPO}/PATH/TO/VAL/DATASET"
  # or use absolute paths
  VAL_CSV_PATH: /PATH/TO/VAL/DATASET/file_name.csv
  VAL_IMG_PATH: /PATH/TO/VAL/DATASET
  
  # flag to make data augmentation on validation images every epoch
  VAL_AUGMENTATION: false
  
  # flag to shuffle validation images every epoch
  VAL_SHUFFLE: false
```
------------------------------
The next section is about Genetic Algorithm configuration:
```yaml
# Genetic Algorithm configuration
GA:
  #flag to continue the last GA run for a few more generations
  #If true, read the last pop from csv file and continue
  #If false, restart the GA
  CONTINUE_EXEC: false
  
  # Seed to be used in GA execution
  # If empty, the seeds that will be used is 0
  SEED: 0
  
  # Maximum number of generations
  NRO_MAX_GEN: 100  
  
  # Population size
  POPULATION_SIZE: 100

  # Crossover rate
  CROSSOVER_RATE: 0.6
  
  # Mutation rate
  MUTATION_RATE: 0.01
  
  # Number of individuals to be selected by elitism
  # If 0, elitism will not be used
  ELITISM: 2
  
  # flat to activates the hypermutation function
  # hypermutation is active if the average of the best 
  # fitness of the last HYPERMUTATION_CYCLE_SIZE generations 
  # is less than or equal to the average of the previous 
  # HYPERMUTATION_CYCLE_SIZE
  HYPERMUTATION: false
  
  # Cycle size considered in hypermutation
  HYPERMUTATION_CYCLE_SIZE: 2

  # Number to be multiplied by the mutation rate during 
  # the hypermutation cycle
  HYPERMUTATION_RATE: 3
  
  # flat to activates the hypermutation function
  # If true, each generation IMIGRATION_RATE percent 
  # of the population is replaced by new random individuals.
  # elitist individuals cannot be replaced
  RANDOM_IMIGRANTS: false

  # Percentage of population to be replaced
  IMIGRATION_RATE: 0.05  
  
  # Number of individuals selected by selection per tournament
  TOURNAMENT_SIZE: 3

   # Genetic Algorithm search space configuration
  SEARCH_SPACE:

    # Padding option to be used by Conv layers
    PADDING: ['valid', 'same']

    # Maximum number of dense layers
    MAX_DENSE_LAYERS: 5

    # Maximum number of cnn layers
    MAX_CNN_LAYERS: 5

    # Amount of neurons in the dense layer
    UNITS: [20, 50, 80, 100, 150, 200, 250, 500]
    
    # Number of filters to be used by Conv layers
    FILTERS: [4, 5, 6, 7, 8, 9]
  
    # Kernel size to be used by Conv layer
    KERNEL_SIZE: [2, 3, 4, 5, 6]

    # Pooling size to be used by Pooling layer
    POOL_SIZE: [2, 3, 4, 5]

    # Batch size to be used during the model train
    BATCH_SIZE: [3, 4, 5, 6, 7, 8]
  
    # Number of epochs to train the model
    EPOCHS: [30, 45, 60, 90, 120, 150, 200]
  
    # Dropout rate
    DROPOUT: [0.0, 0.1, 0.2, 0.3, 0.4]

    # Optimizer used in training the model
    OPTIMIZER: ['Adam', 'SGD']
    
    # Learning rate used in training the model.
    LEARNING_RATE: [0.0001, 0.001, 0.01, 0.05, 0.1]
    
    # Standardization method used to scale the csv data
    SCALER: ['Standard', 'MinMax']
    
    # Activation function to use in the Dense layers.
    ACTIVATION_DENSE: ['relu', 'tanh']
    
    # Activation function to use in the CNN layers.
    ACTIVATION_CNN: ['relu', 'tanh']
    
    # Regularizer function applied to the kernel weights matrix.
    KERNEL_REGULARIZER: [None]
    
    # Initializer for the kernel weights matrix.
    KERNEL_INITIALIZER: ['glorot_uniform']
    
    # Regularizer function applied to the output of the layer (its activation). 
    ACTIVITY_REGULARIZER: [None]
    
    # Regularizer function applied to the bias vector. 
    BIAS_REGULARIZER: [None]
```

------------------------------
The following section is about model construction:
```yaml
# Model experiment configuration
MODEL:
  # Define the architecture to be used
  # Available options for architecture are MLP, CNN and CNN/MLP
  ARCHITECTURE: "MLP"
  
  # The backbone define the specific ARCHITECTURE to be used
  # In Classification tasks on top of this backbone will be created 3 Dense
  # layers and the final layer with number of neurons equal number of classes.
  # Available options for backbone are:
  #
  # GASearch -> the backbone is defined by the GA. This option not use pre trained models.
  #
  # For CNN and CNN/MLP, is used pre-trained models like: VGG16, VGG19, ResNet50, ResNet101, 
  # ResNet152, ResNet50V2, ResNet101V2,ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet, 
  # MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge
  #
  # For MLP is created a MLP with 3 Dense layers and the final layer with number of neurons equal number of classes.
  #
  #
  # In Segmentation tasks this backbone will be used as an encoder, the final
  # arange depends of segmentation architecture
  BACKBONE: "GASearch"

  # flat to activates feature selection
  FEATURE_SELECTION: false  

  # Define input shape of experiment
  # For pre-trained models, should be with 3 channels: [, , 3]
  INPUT_SHAPE: [150, 150, 3]
  
  # Define the Dropout rate to be used by the model
  # Is used only when the BACKBONE is not "GASearch"
  DROPOUT: 0.2
```

------------------------------
The next section is about solver hyperparameters.
```yaml
# Solver experiment configuration
SOLVER:
  # Loss function to be used
  # The values follow the keras API
  LOSS: "categorical_crossentropy"

  # List of Metric functions to be used
  # The values follow the keras API
  METRICS: ["categorical_accuracy"]
  
  # flat to activates cross validation
  CROSS_VALIDATION: true

  # Number of folds to be user by the cross validation method,
  # if CROSS_VALIDATION is activated
  CROSS_VALIDATION_FOLDS: 10

  # Test fold size to be used on train test split
  # The train test split is performed to evaluate individuals of the GA, 
  # where the training dataset is divided into training and testing, 
  # where the random seed is determined by the generation id, that is, 
  # each generation the training and test folds change
  TEST_SIZE: 0.2
  
  # Batch size to be used during the model train
  # Is used only when the BACKBONE is not "GASearch"
  BATCH_SIZE: 128

  # Number of epochs to train the model
  # Is used only when the BACKBONE is not "GASearch"
  EPOCHS: 100

  # Optimizer used in training the model
  # Is used only when the BACKBONE is not "GASearch"
  OPTIMIZER: "Adam"

  # Learning rate used in training the model.
  # Is used only when the BACKBONE is not "GASearch"
  BASE_LEARNING_RATE: 0.0001
```

------------------------------
The last option is about the path to store trainned model.
```yaml
# path to save checkpoints files
OUTPUT_DIR: ./NNGA_output
```