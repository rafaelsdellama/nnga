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
# Dataset experiment configuration
DATASET:
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
  NRO_MAX_EXEC: 1
  SEED: []
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
  FEATURE_SELECTION: false    

# Model experiment configuration
MODEL:
  ARCHITECTURE: "MLP"
  INPUT_SHAPE: [200, 200, 3]

# Solver experiment configuration
SOLVER:
  LOSS: "categorical_crossentropy"
  METRICS: ["categorical_accuracy"]

OUTPUT_DIR: ./NNGA_output
```

#### A tour to all experiments options

Let's go to take a closer look of all options available in an experiment.  

The first section is about datasets: 
```yaml
# Dataset experiment configuration
DATASET:
  #flag to preserve the aspect ratio when the image is resized to INPUT_SHAPE
  #If true, resize the image while maintaining the aspect ratio, adding a black background
  #If false, resize the image to INPUT_SHAPE without worrying about maintaining the aspect ratio
  PRESERVE_IMG_RATIO: true

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
  #flag to continue the last GA run for a few more generations
  #If true, read the last pop from csv file and continue
  #If false, restart the GA
  CONTINUE_EXEC: false

  # Number of GA executions changing the random seed 
  NRO_MAX_EXEC: 1
  
  # Seeds to be used in executions
  # The number of seeds should be equal to NRO_MAX_EXEC
  # If empty, the seeds that will be used are range(0, NRO_MAX_EXEC)
  SEED: []
  
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

  # flat to activates feature selection
  FEATURE_SELECTION: false    
```

------------------------------
The following section is about model construction:
```yaml
# Model experiment configuration
MODEL:
  # Define the architecture to be used
  # Available options for architecture are MLP, CNN and CNN/MLP
  ARCHITECTURE: MLP

  # Define input shape of experiment
  INPUT_SHAPE: [200, 200, 3]
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
```

------------------------------
The last option is about the path to store trainned model.
```yaml
# path to save checkpoints files
OUTPUT_DIR: ./NNGA_output
```