# main run file
import argparse
import matplotlib.pyplot as plt

from utils import show_grid, make_grid

parser = argparse.ArgumentParser()
parser.add_argument('--backend', help="Keras or PyTorch backends", type=str)
parser.add_argument('-n', '--latent_dim', help="Latent-space dimensionality", type=int, default=2)
parser.add_argument('-bs', '--batch_size', help="Training batch size", type=int, default=1024)
parser.add_argument('-e', '--epochs', help="Number of training epochs", type=int, default=20)
parser.add_argument('-hd', '--hidden_dim', help="Number of units in the hidden layers", type=int, default=512)
args = parser.parse_args()

if args.backend == None:
    raise ValueError('Backend must be set to either keras or pytorch! Use the --backend option')
elif args.backend.lower() == 'keras':
    from vae.keras import load_data, train_model
    backend = 'keras'
elif args.backend.lower() == 'pytorch':
    import torch
    from torch.autograd import Variable
    from vae.pytorch import load_data, train_model
    backend = 'pytorch'
else:
    raise ValueError('Backend must be set to either keras or pytorch, got %s' %args.backend)

# train
n = args.latent_dim
epochs = args.epochs
batch_size = args.batch_size

model, enc, dec = train_model(n=n, epochs=epochs, batch_size=batch_size)


if backend == 'keras':
    (x, x_val), y = load_data()
    if n == 2:
        # display latent space
        x_enc = enc.predict(x, batch_size=batch_size)[0]
        plt.scatter(x_enc[:, 0], x_enc[:, 1], c=y[:len(x)])
        plt.colorbar()
        plt.show()

    # display some decoded images
    x_pred = model.predict_on_batch(x[:batch_size])
    x_pred = x_pred.reshape(batch_size, 28, 28)

    x = x.reshape(len(x), 28, 28)
    x_pred_grid = make_grid(x_pred)
    x_grid = make_grid(x)


if backend == 'pytorch':
    train, test = load_data(batch_size=batch_size)
    if n == 2:
        # display latent space
        for i, (x, y) in enumerate(train):
            x = Variable(x)
            if torch.cuda.is_available():
                x = x.cuda()
            data = enc(x)[0]
            data = data.data.cpu().numpy()
            label = y.cpu().numpy()
            plt.scatter(data[:,0], data[:,1], c=label)
        plt.colorbar()
        plt.show()

    # display some decoded images
    x, _ = next(iter(train)) # get a batch of data
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()

    x_pred = model(x)
    x_pred = x_pred.data.cpu().numpy()
    x_pred = x_pred.reshape(batch_size, 28, 28)
    x = x.data.cpu().numpy()

    x = x.reshape(len(x), 28, 28)
    x_pred_grid = make_grid(x_pred)
    x_grid = make_grid(x)


fig, ax = plt.subplots(1, 2, dpi=150)

ax[0].imshow(x_grid, cmap='Greys_r')
ax[0].set_title('Original')

ax[1].imshow(x_pred_grid, cmap='Greys_r')
ax[1].set_title('Decoded')

for axes in ax:
    axes.set_axis_off()

plt.show()