import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

import equinox as eqx

# torch.Size([1, 105, 105])

class CNN(eqx.Module):
    layers: list

    def __init__(self, key, channels):

        key1, key2, key3, key4, key5 = jax.random.split(key,5)

        self.channels = channels

        self.layers = [
            eqx.nn.Conv2d(self.channels, self.channels, kernel_size=3, key=key1),
            eqx.nn.BatchNorm(self.channels,channelwise_affine=True, axis_name="batch"),
            jax.nn.elu,
            eqx.nn.MaxPool2d(kernel_size=2),
            eqx.nn.Conv2d(self.channels, self.channels, kernel_size=3, key=key2),
            eqx.nn.BatchNorm(self.channels,channelwise_affine=True, axis_name="batch"),
            jax.nn.elu,
            eqx.nn.MaxPool2d(kernel_size=2),
            eqx.nn.Conv2d(self.channels, self.channels, kernel_size=3, key=key3),
            eqx.nn.BatchNorm(self.channels,channelwise_affine=True, axis_name="batch"),
            jax.nn.elu,
            eqx.nn.MaxPool2d(kernel_size=2),
            eqx.nn.Conv2d(self.channels, self.channels, kernel_size=3, key=key4),
            eqx.nn.BatchNorm(self.channels,channelwise_affine=True, axis_name="batch"),
            jax.nn.elu,
            eqx.nn.MaxPool2d(kernel_size=2),
            jnp.ravel
        ]
    
    def __call__(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        normalization = jnp.sqrt(x.shape[-1])
        return x / normalization