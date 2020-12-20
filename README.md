# Bank Credit Score Prediction

This contains a neural network implementation which predict a credit score for bank customers. 

## Files

>archive.zip

This file contains the dataset. It compose of two **csv** file, which are **train set** and **test set**.


>get_data.py

This file fetches the data from `<archive.zip>` and returns two **csv** file, one first one of this files is the **train set** and second one is the **test set**. It creates a *dataset* directory to the current directory where file is. The *dataset* directory contains those csv files and the *archive.zip* that renamed with *dataset.zip*. If there is already a file which name is *dataset.zip* in the dataset directory, it will asked a new name for this *archive.zip* file to create it.

>prepare_data.py

This file prepares both the **train set** and **test set** which downloaded via *get_data.py* for the model. It returns a variable whose type is dictionary.
This dictionary contains a **train set**, a **validation set** and a **test set**, and also *its labels*.

>neural_network.py

This file is the main file of the project. It contains an ANN architucture which compose of three dense layers. It had been created with `<keras' sequential api>`. It fetches data via *prepare_data.py* file and trains the model.

>Note: As you may expect that all files must be in the same directory.