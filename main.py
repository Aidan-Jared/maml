import optax
import jax
import jax.numpy as jnp
import equinox as eqx
import tqdm as tqdm
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

from sampleTask import FCS_loader

from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy

from torchvision import datasets, transforms
from torch import manual_seed

from model.cnn import CNN

from metalearners.anil import ANIL, SAM_ANIL
from metalearners.imaml import IMAML

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'

SEED = 42
KEY = jax.random.PRNGKey(SEED)

manual_seed(seed=SEED)
               
def main():

    device = jax.devices('gpu')[0]
    jax.config.update('jax_default_device', device)


    normalize_data = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize(84),
        transforms.RandomRotation(15),
        # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.Omniglot(
        "Omniglot",
        transform= normalize_data,
        download=True,
        background=True
    )

    key, subkey = jax.random.split(KEY)

    # device = jax.devices('cpu')[0]
    # jax.config.update('jax_default_device', device)

    sampler = FCS_loader(dataset, subkey, batch_size=4)

    shape = dataset[0][0].shape

    model = CNN(key=key, channels= shape[0], width=shape[1], height=shape[2])

    # maml = IMAML(alpha=1e-4, beta=1e-4, lambda_=1., cg_steps=10, cg_damping=1., grad_clip=5., inner_steps=50)

    anil = SAM_ANIL()

    model = eqx.nn.inference_mode(model, value=True)
    # model = maml.train(model, sampler, subkey)

    model = anil.train(model, sampler, subkey, epochs=5000)


if __name__ == "__main__":
    main()
