# vae-keras-pytorch

Variational autoencoder reproduced in keras and pytorch.

# Overview

A variational autoencoder allows for the encoding of high-dimensional data into a low dimensional latent space. Apart from serving as a dimensionality reduction tool, one can also perform latent space exploration to produce authentic samples that do not exist in the original training data.

# Dependancies

* matplotlib
* keras
* pytorch

# Usage

To train a variational auto encoder on the MNIST handwritten number dataset using the keras backend, run

```console
$ python main.py --backend keras
```

When using the pytorch backend, run

```console
$ python main.py --backend pytorch
```

Apart from the backend, the `main.py` file accepts the following options

```console
-n,   --latent_dim  Latent-space dimensionality, default = 2
-bs,  --batch_size  Training batch size, default = 1024
-e,   --epochs      Number of training epochs, default = 20
-hd,  --hidden_dim  Number of units in the hidden layers, default=512
```

# Output

Once training is complete, the `main.py` file will produce a figure as output, showing an example of the original training data with the corresponding decoded output of the variational autoencoder. 


If the latent space dimensionality is 2, `main.py` will also produce a figure displaying the latent space of the training data, color coding each point to correspond with the training label (e.g. a handwritten 2 has a label 2).
