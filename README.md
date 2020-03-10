# **N**eural **N**etwork optimized by **G**enetic **A**lgorithms - (NNGA)
----
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

## Image Dataset Directory Structure
----
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

## Models
----

### Multilayer Perceptron with Feature Selection optimized by Genetic Algorithms (MLP)
----
![mlp](/doc/images/mlp.png)

### Convolutional Neural Network optimized by Genetic Algorithms (CNN)
----
![cnn](/doc/images/cnn.png)

### Hybrid CNN-MLP Model with Feature Selection optimized by Genetic Algorithms (CNN/MLP)
----
![cnn_mlp](/doc/images/cnn_mlp.png)

## Installation
----

## Documentation
----
Please, read the [getting-started.md](/getting-started.md) for the basics.

## License
----
MIT
