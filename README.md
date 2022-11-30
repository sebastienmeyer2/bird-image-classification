# Bird Image Classification

The goal of this project is to study and apply machine learning/artificial intelligence techniques to predict the categories of bird images. We will evaluate our methods on a list of mystery images. The dataset contains training, validation and test images.

The data set will eventually be publicly available on the [RecVis website](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip).

Please refer to the following sections for more information about the package usage:

1. [Our results](#our-results)
2. [Installation](#installation-instructions)
3. [Description](#package-description)
4. [Usage via command lines](#package-usage)
5. [Documentation](#documentation)

## Our results

A brief summary of our results is available in our report under *report/report.pdf*. Below, we only give a summary table of the test accuracy of different models.

| Model                                               | Test accuracy |
| --------------------------------------------------- | ------------- |
| CNN                                                 | 0.17419       |
| Transfer Learning (TL)                              | 0.61290       |
| TL + Data Augmentation & Hyperparameter Tuning      | 0.85806       |

## Installation instructions

In order to use our package and run your own experiments, we advise you to set up a virtual environment. The package has been tested under Python version 3.9.10, you will also need the virtualenv package:

    pip3 install virtualenv

Then, you can create a virtual environment and switch to it with the following commands:

    python3 -m venv myvenv
    source myvenv/bin/activate (Linux)
    myvenv/Scripts/Activate.ps1 (Windows PowerShell)

All the needed packages are listed in the requirements file, you can install them with:

    pip3 install -r requirements.txt

This file expects you to have PyTorch version 1.13 with CUDA>=11.7 installed on your machine. If it is not the case, install the version via command line or install your preferred version locally then remove the lines related to torch in the requirements.txt file and use the command again.

## Package description

Below, we give a brief tree view of our package.

    .
    ├── doc  # contains a generated documentation of src/ in html
    ├── report  # contains our complete report in pdf format
    ├── src  # source code
    |   ├── engine
    |   |   ├── models
    |   |   |   ├── __init__.py
    |   |   |   ├── cnn.py  # custom and transfer learning convolutional networks
    |   |   |   └── transformer.py  # transfer learning ViT
    |   |   ├── __init__.py
    |   |   ├── gridsearch.py
    |   |   ├── hub.py  # to create models
    |   |   └── training.py  # to start training
    |   ├── utils
    |   |   ├── __init__.py
    |   |   ├── data_preparation.py  # utilitary function to preprocess data
    |   |   └── seed_handler.py
    |   ├── __init__.py
    |   └── main.py  # main file to run gridsearch
    ├── README.md
    └── requirements.txt  # contains the necessary Python packages to run our files

## Package usage

### Downloading the data

The data set will eventually be publicly available on the [RecVis website](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). If it is the case, you can place the train_images, val_images and test_images directories into a *data/* folder.

### Gridsearch

Then, you can use the *src/main.py* file to try multiple gridsearch and models. The command is as follows:

    python3 src/main.py [options]

- `--seed`: Seed to use everywhere for reproducbility. Default: 42.
- `--models-names`: Choose models names. Available models: "baseline", "evolved", "transfered" and "google".
- `--data-path`: Path to the directory where the data is stored. Default: "data/".
- `--trials`: Choose the number of gridsearch trials. Default: 25.

## Documentation

A complete documentation is available in the *doc/src/* folder. If it is not
generated, you can run from the root folder:

    python3 -m pdoc -o doc/ --html --config latex_math=True --force src/

Then, open *doc/src/index.html* in your browser and follow the guide!

## Acknowledgments

Adapted from [Rob Fergus and Soumith Chintala](https://github.com/soumith/traffic-sign-detection-homework)
Adaptation done by [Gul Varol](https://github.com/gulvarol)
