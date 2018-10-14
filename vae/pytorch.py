# pytorch implementation of a variational autoencoder

import os

import torch
import torch.nn as nn 
import torch.nn.functional as f
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

cuda = torch.cuda.is_available()
MNIST_ROOT = os.path.expanduser('~/.torch/MNIST/')


def vae_model(n=2, hidden_dim=512):
    """Builds a variational autoencoder.

    Params
    ------

    n : int
        The dimensionality of the autoencoder

    hidden_dim: int
        Dimensionality of the decoder hidden layers

    Returns
    -------

    model: Pytorch model
        A pytorch model of a encoder-decoder model

    enc_model: Pytorch model
        A pytorch model of the vae encoder

    dec_model: Pytorch model
        A pytorch model of the vae decoder
    """
    enc = Encoder(output_dim=n, hidden_dim=hidden_dim)
    dec = Decoder(input_dim=n, hidden_dim=hidden_dim)

    if cuda:
        enc = enc.cuda()
        dec = dec.cuda()

    encdec = EncDec(enc, dec)

    return encdec, enc, dec


def train_model(n=2, hidden_dim=512, epochs=10, batch_size=32):
    """Trains the VAE model

    Params
    ------

    n : int
        The latent-space dimensionality of the autoencoder

    hidden_dim: int
        Dimensionality of the hidden layers

    epochs: int
        Number of epochs to train on

    batch_size: int
        Batch size during training

    Returns
    -------

    trained: 3-tuple
        Contains the encoder-decoder model as (encoder-decoder, encoder, decocer).
    """
    train_loader, test_loader = load_data(batch_size=batch_size)
    encdec, encoder, decoder = vae_model(n=n, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    for epoch in range(epochs):
        epoch_loss = 0
        for x_train, _ in train_loader:
            # init optimizer
            optimizer.zero_grad()
            
            x_train = Variable(x_train)
            if cuda:
                x_train = x_train.cuda()
            
            # get latent components and predict image
            z_mean, z_var = encoder(x_train)
            x_pred = decoder(z_mean)
            
            # compute loss and optimize
            loss = kl_loss(z_mean, z_var) + im_loss(x_train, x_pred)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data[0]/len(train_loader)
            
        if epoch % 1 == 0:
            # compute validation loss
            val_loss = 0
            for x_test, _ in test_loader:
                x_test = Variable(x_test)
                if cuda:
                    x_test = x_test.cuda()
                # get latent components and predict image
                z_mean, z_var = encoder(x_test)
                x_pred = decoder(z_mean)

                # compute loss and optimize
                loss = kl_loss(z_mean, z_var) + im_loss(x_test, x_pred)
                val_loss += loss.data[0]/len(test_loader)
                
            print('Epoch %d - Loss: %.2f - Val Loss: %.2f' %(epoch+1, epoch_loss, val_loss))

    return encdec, encoder, decoder


# utility and loss functions

def load_data(batch_size=32):
    train_dataset = MNIST(MNIST_ROOT, train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(MNIST_ROOT, train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def kl_loss(z_mean, z_var):
    """Computes the KL divergence between N(0,1) and N(z_mean, z_std)"""
    # assumes z_var is a log variable
    return 0.5*torch.mean(-z_var - 1 + torch.exp(z_var) + z_mean**2)

def im_loss(x_true, x_pred):
    x_true = x_true.view(-1, 1*28*28)
    x_pred = x_pred.view(-1, 1*28*28)
    return 28 * 28 * torch.mean((x_true - x_pred)**2)


# Classes

class Encoder(nn.Module):
    
    def __init__(self, output_dim, hidden_dim=512):
        """Create a 2 layer convulutional encoder."""
        super(Encoder, self).__init__()
        self.output_dim = output_dim
        self.hidden1 = nn.Linear(28*28*1, 2*hidden_dim)
        self.hidden2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, output_dim)
        self.linear_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view((-1, 28*28*1))
        x = self.hidden1(x)
        x = f.relu(x)
        x = self.hidden2(x)
        x = f.relu(x)
        z_mean = self.linear_mean(x)
        z_log_var = self.linear_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=512):
        """Create a 2 layer decoder given a vector of dimension 'input_dim'."""
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.output = nn.Linear(2*hidden_dim, 28*28*1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = f.relu(x)
        x = self.linear2(x)
        x = f.relu(x)
        x = self.output(x)
        x = f.sigmoid(x)
        x = x.view((-1, 28, 28, 1))
        return x

class EncDec(nn.Module):

    def __init__(self, encoder, decoder):
        """Create a 2 layer decoder given a vector of dimension 'input_dim'."""
        super(EncDec, self).__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, x):
        x = x.view(-1, 28, 28, 1)
        z_mean, _ = self.enc(x)
        x = self.dec(z_mean)
        return x