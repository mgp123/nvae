# NVAE Implementation
<p align="center">
    <img src="images/cherry_0.4_big_logistic_mixture20latentnoflows.png" width=350>
</p>

This is a close enough implementation of *NVAE. A Deep Hierarchical Variational Autoencoder*.

Some differences with the original implementation:

- No spectral regularization
- In the discrete logistic mixture each pixel channel has its own set of selectors instead of one per pixel

## Training
To train a new model you simply need to do:

    python3 train.py <architecure name>

with ```<architecure name>``` being the name of one of the ```.yaml``` files in ```model_configs```

This files contains all the architecture hyperparameters as well as training parameters such as batch size or number of epochs between checkpoints.
You can also resume training from the last checkpoint using the same command. 
The script automatically finds the last checkpoint before starting trainning.

## Creating a model configuration