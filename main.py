# main run file
import matplotlib.pyplot as plt

from utils import show_grid, make_grid
from vae.keras import train_model, load_data

# train
(x, x_val), y = load_data()
batch_size = 1024
n = 10

(model, enc, dec), history = train_model(n=n, epochs=20, batch_size=batch_size)

# display latent var
if n == 2:
    x_enc = enc.predict(x, batch_size=batch_size)[0]
    plt.scatter(x_enc[:, 0], x_enc[:, 1], c=y[:len(x)])
    plt.colorbar()
    plt.show()

# display decoded
x_pred = model.predict_on_batch(x[:batch_size])
x_pred = x_pred.reshape(batch_size, 28, 28)
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