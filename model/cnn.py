import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Int

import equinox as eqx

# torch.Size([1, 105, 105])

class CNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout
    dropout4: eqx.nn.Dropout

    def __init__(self, key, channels = 1, width = 105, height = 105, n_way = 5, dropout = .1):

        key1, key2, key3, key4, key5= jax.random.split(key,5)


        w1 = (width - 4) / 1 + 1
        h1 = (height - 4) / 1 + 1

        w2 = int((w1 - 4) / 1 + 1)
        h2 = int((h1 - 4) / 1 + 1)

        self.conv1 = eqx.nn.Conv2d(channels, 32 , kernel_size=4, key=key1)
        
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=4, key=key2)
        self.fc1 = eqx.nn.Linear(64 * w2 * h2, 256 ,key=key3)
        self.fc2 = eqx.nn.Linear(256, 32, key=key4)
        self.fc3 = eqx.nn.Linear(32, n_way, key=key5)

        self.dropout1 = eqx.nn.Dropout(p=dropout)
        self.dropout2 = eqx.nn.Dropout(p=dropout)
        self.dropout3 = eqx.nn.Dropout(p=dropout)
        self.dropout4 = eqx.nn.Dropout(p=dropout)
    
    @eqx.filter_checkpoint
    def __call__(self, x: Float[Array, " channels width height"], key) -> Float[Array, "n_way"]:

        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)

        x = jax.nn.elu(self.conv1(x))
        x = self.dropout1(x, key=subkey1)

        x = jax.nn.elu(self.conv2(x))
        x = self.dropout2(x, key=subkey2)

        x = jnp.ravel(x)
        x = jax.nn.elu(self.fc1(x))
        x = self.dropout3(x, key=subkey3)

        x = jax.nn.elu(self.fc2(x))
        x = self.dropout4(x, key=subkey4)

        x = self.fc3(x)

        return jax.nn.log_softmax(x)