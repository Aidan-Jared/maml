import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

import equinox as eqx

# torch.Size([1, 105, 105])

class CNN(eqx.Module):
    layers: list

    def __init__(self, key, channels = 1, width = 105, height = 105, n_way = 5):

        key1, key2, key3, key4, key5 = jax.random.split(key,5)

        w1 = (width - 4) / 1 + 1
        h1 = (height - 4) / 1 + 1

        w2 = int((w1 - 4) / 1 + 1)
        h2 = int((h1 - 4) / 1 + 1)

        self.layers = [
            eqx.nn.Conv2d(channels, 15 * channels, kernel_size=4, key=key1),
            eqx.nn.Conv2d(15 * channels, 30 * channels, kernel_size=4, key=key2),
            jax.nn.elu,
            jnp.ravel,
            eqx.nn.Linear(channels * 15 * 2 * w2 * h2, 256 ,key=key3),
            jax.nn.tanh,
            eqx.nn.Linear(256, 32, key=key4),
            jax.nn.elu,
            eqx.nn.Linear(32, n_way, key=key5),
            jax.nn.log_softmax
        ]
    
    def __call__(self, x: Float[Array, " channels width height"]) -> Float[Array, "n_way"]:
        
        for layer in self.layers:
            x = layer(x)

        return x