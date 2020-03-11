# **N**eural **N**etwork optimized by **G**enetic **A**lgorithms - (NNGA)

Neural Network optimized by Genetic Algorithms - (NNGA) is a library for deep model training, by Backpropagation, for data classification. The adjustment of the model parameters and the selection of resources are done by a genetic algorithm.
The model accepts two data entry sources:
1. Images
    2.1 The format of the images should be:
    * Windows bitmap (bmp)
    * Portable image formats (pbm, pgm, ppm)
    * Sun raster (sr, ras)
    * JPEG (jpeg, jpg, jpe)
    * JPEG 2000 (jp2)
    * TIFF files (tiff, tif)
    * Portable network graphics (png)
2. CSV file
    1.1. The sample class should be in the column with label "class"
    1.2 If the two data sources are used simultaneously, the CSV file should be a column with label "id" containing the corresponding image name (without extension)

## Installation
```bash
git clone https://github.com/rafaelsdellama/nnga.git
cd nnga
pip install .
```

## Models

### Multilayer Perceptron with Feature Selection optimized by Genetic Algorithms (MLP)
![mlp](/doc/images/mlp.png)

The Genetic Algorithm chromosome that encodes this model is shown below:

![mlp](/doc/images/indiv_mlp.jpg)

### Convolutional Neural Network optimized by Genetic Algorithms (CNN)
![cnn](/doc/images/cnn.png)

The Genetic Algorithm chromosome that encodes this model is shown below:

![mlp](/doc/images/indiv_cnn.jpg)

### Hybrid CNN-MLP Model with Feature Selection optimized by Genetic Algorithms (CNN/MLP)
![cnn_mlp](/doc/images/cnn_mlp.png)

The Genetic Algorithm chromosome that encodes this model is shown below:

![mlp](/doc/images/indiv_cnn_mlp.jpg)

## Parameters optimized by the Genetic Algorithm
### Optimized parameters for each layer of the Convolutional  Neural Network:

| Parameter | activate | filter | kernel_size | padding | batchNormalization | maxPool | pool_size | dropout |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |  :---: | 
| Values | True <br/> False | The number of filters is defined by the function: filters = 2^i, where i is a value of the sequence of integer numbers from 4 to 8, increment by 1 | A value of the sequence of integer numbers from 2 to 8, increment by 1  | valid <br/> same | True <br/> False | True <br/> False | A value of the sequence of integer numbers from 2 to 6, increment by 1 | A value of the sequence of integer numbers from 0 to 0.4, increment by 0.1 |
| Description | True: this layer is added to the model  <br/><br/> False: this layer is not added to the model  | Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution) | An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions. | Valid: means "no padding" <br/><br/> Same:  results in padding the input such that the output has the same length as the original input | True: Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1 <br/><br/> False: Does not do anything | True: Applies the max pooling operation <br/><br/> False: Does not do anything	| Integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions. |	Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. |

### Optimized parameters for each layer of the Multilayer Perceptron:

| Parameter | activate | units | dropout |
| :---: | :---: | :---: | :---: |
| Values | True <br/> False | A value of the sequence of integer numbers from 20 to 100, increment by 5 | A value of the sequence of integer numbers from 0 to 0.4, increment by 0.1 |
| Description | True: this layer is added to the model  <br/><br/> False: this layer is not added to the model" | The amount of neurons in the layer. | Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. |

### Optimized model parameters:

| Parameter | activation_conv | activation_dense | batch_size | epochs | optimizer | learning_rate | scaler |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Values | relu <br/> tanh | relu <br/> tanh  | The number of filters is defined by the function: filters = 2^i, where i is a value of the sequence of integer numbers from 3 to 7, increment by 1	 |  A value of the sequence of integer numbers from 20 to 200, increment by 5 | adam <br/> SGD <br/> RMSprop <br/> Adagrad <br/> Adadelta <br/> Adamax <br/> Nadam  | 0.001 <br/> 0.0025 <br/> 0.005 <br/> 0.0075 <br/> 0.01 <br/> 0.025 <br/> 0.05 <br/> 0.075 <br/> 0.1 <br/> 0.25 <br/> 0.5 <br/> 0.75 <br/> 1.0 <br/> 1.1 | Standard <br/> MinMax | 
| Description | Activation function to use in the CNNs layer. | Activation function to use in the Denses layer. | Number of samples per gradient update. | Number of epochs to train the model. | Optimizer used in training the model. | Learning rate used in training the model. | Standardization method used to scale the data |

| Parameter | kernel_regularizer | kernel_initializer | activity_regularizer | bias_regularizer |
| :---: | :---: | :---: | :---: | :---: |
| Values | None  <br/> l1 <br/> l2 <br/> l1_l2 | Zeros <br/> Ones <br/> RandomNormal <br/> RandomUniform <br/> TruncatedNormal <br/> VarianceScaling <br/> Orthogonal <br/> lecun_uniform <br/> glorot_normal <br/> glorot_uniform <br/> he_normal <br/> lecun_normal <br/> he_uniform | None  <br/> l1 <br/> l2 <br/> l1_l2 | None  <br/> l1 <br/> l2 <br/> l1_l2 |
| Description | Regularizer function applied to the kernel weights matrix. <br/> https://keras.io/regularizers/ |  Initializer for the kernel weights matrix. <br/> https://keras.io/initializers/ | Regularizer function applied to the output of the layer (its activation). <br/> https://keras.io/regularizers/ | Regularizer function applied to the bias vector. <br/> https://keras.io/regularizers/ |


## Image Dataset Directory Structure

There is a standard way to lay out your image data for modeling.
After you have collected your images, you must sort them first by dataset, such as train, test, and validation, and second by their class.
Under each of the dataset directories, we will have subdirectories, one for each class where the actual image files will be placed.

The directory structure should look like this:

    data/train/
        ├── class_1/
    		├── image0001.jpg
    		├── ...
        ├── ...
        ├── class_N/
    		├── image0001.jpg
    		├── ...
    data/val/
        ├── class_1/
    		├── image0001.jpg
    		├── ...
        ├── ...
        ├── class_N/
    		├── image0001.jpg
    		├── ...

## Documentation
Please, read the [getting-started.md](/getting-started.md) for the basics.

## License
MIT
