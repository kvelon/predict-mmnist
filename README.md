# Video Prediction Project for Moving-MNIST Data

This repository contains the code required to 1) download moving-mnist data, 2) train three different deep learning architectures on the data, and 3) plot the frames generated by the trained models.

## Downloading data
MNIST dataset is already part of Pytorch's dataset class. The Pytorch Lightning Dataset class defined in *data/data_classes.py* handles the generation of moving mnist video sequences. No further scripts are required to download additional data.

## Training
The python scripts prefixed with *train_* are for training the various models. The hyperparameters for training are defined in those scripts. In your favourite environment (i.e. conda, Docker), run 
```python train_predrnn.py```
to train a PredRNN model. Metrics are automatically logged with a Tensorboard logger.

## Plotting
The Tensorboard logger already contains a sample plot of the predicted frames. If you wish to make further plots, you can use the *plot_5to5_plot1.ipynb*
 notebook to do so. You will need to modify the checkpoint path to load the trained paramaters and weights.
