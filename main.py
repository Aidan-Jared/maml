import optax
import jax
import jax.numpy as jnp
import equinox as eqx
import tqdm as tqdm
import random
from jaxtyping import Array, Float, Int, PyTree

import torchvision

from model.cnn import CNN
from metalearners.maml import MAML

seed = 42
key = jax.random.PRNGKey(seed)

def sample_task(dataset, n_way=5, k_shot=1, q_query=15):

    class_to_indices = {}

    for idx, (_, label) in enumerate(dataset):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    selected_classes = random.sample(list(class_to_indices.keys()), n_way)

    support_set = []
    query_set = []

    for new_label, original_class in enumerate(selected_classes):
        indicies = class_to_indices[original_class]
        random.shuffle(indicies)

        support_set.extend([(idx, new_label) for idx in indicies[:k_shot]])
        query_set.extend([(idx, new_label) for idx in indicies[k_shot:k_shot + q_query]])
    return support_set, query_set

def main():
    normalize_data = torchvision.transforms.Compose(
        [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = torchvision.datasets.Omniglot(
        "Omniglot",
        transform= normalize_data,
        download=True,
        background=True
    )

    model = CNN(key=key, channels=1)

    metalearner = MAML(model=model)

    


if __name__ == "__main__":
    main()
