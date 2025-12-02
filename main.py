import optax
import jax
import jax.numpy as jnp
import equinox as eqx
import tqdm as tqdm
import random
from jaxtyping import Array, Float, Int, PyTree

from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy

import torchvision

from model.cnn import CNN

from functools import partial

seed = 42
key = jax.random.PRNGKey(seed)

class Sample_Task:
    def __init__(self, dataset, key, n_ways=5, k_shot=1, q_query = 15) -> None:
        self.class_to_indices = {}

        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.key=key
        self.n_ways=n_ways
        self.k_shot=k_shot
        self.q_query=q_query

        self.dataset = dataset

    def sample(self):
        selected_classes = jax.random.choice(key=self.key, a= jnp.array(list(self.class_to_indices.keys()), dtype=jnp.int32), shape=(1, self.n_ways), replace=False)[0]
        self.key, _ = jax.random.split(self.key)
        

        support_data = []
        query_data = []

        support_target = []
        query_target = []

        for new_label, original_class in enumerate(selected_classes):
            indicies = self.class_to_indices[original_class.item()]
            
            random.shuffle(indicies)



            support_data.extend([self.dataset[idx][0].numpy() for idx in indicies[:self.k_shot]])
            query_data.extend([self.dataset[idx][0].numpy() for idx in indicies[self.k_shot:self.k_shot + self.q_query]])

            support_target.extend([new_label] * self.k_shot)
            query_target.extend([new_label] * self.q_query)
        
        support_set = (jnp.stack(support_data), jnp.stack(support_target))
        query_set = (jnp.stack(query_data), jnp.stack(query_target))
    
        return support_set, query_set

@eqx.filter_jit
def loss( model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"]) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    loss = cross_entropy(pred_y, y)
    
    return jnp.mean(loss)

@partial(eqx.filter_jit, donate="none")
def inner_loop(
        model: CNN,
        support_set: tuple[Float[Array, "Channels Width Height"], Int],
        query_set: tuple[Float[Array, "Channels Width Height"], Int],
        alpha: Float,
        batch: Int = 5
):
    
    optim = optax.sgd(alpha)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    def make_step(
        model, opt_state
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, support_set[0], support_set[1])
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss_value
    
    inner_losses = []
    for _ in range(batch):
        (model, opt_state), inner_loss = make_step(model, opt_state)
        inner_losses.append(inner_loss)

    outer_loss, outer_grads = eqx.filter_value_and_grad(loss)(model, query_set[0], query_set[1])

    del model, opt_state

    return jnp.mean(jnp.array(inner_losses)), outer_loss, outer_grads

def train(
        model: CNN,
        sampler: Sample_Task,
        alpha: Float = .05,
        beta: Float = .001,
        task_batch: Int = 5,
        inner_batch: Int = 5,
        epochs: Int = 100,
):
    optim_outer = optax.adamw(beta)
    opt_state_outer = optim_outer.init(eqx.filter(model, eqx.is_array))

    for epoch in tqdm.tqdm(range(epochs)):
        accumulated_grads = None
        inner_losses = []
        outer_losses = []
        
        # Process tasks one at a time to save memory
        for _ in range(task_batch):
            support_set, query_set = sampler.sample()
            
            # Process single task
            inner_loss, outer_loss, outer_grads = inner_loop(
                model, support_set, query_set, alpha, inner_batch
            )
            
            # Convert to float immediately to free GPU memory
            inner_losses.append(float(inner_loss))
            outer_losses.append(float(outer_loss))
            
            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = outer_grads
            else:
                accumulated_grads = jax.tree_util.tree_map(
                    lambda acc, new: acc + new,
                    accumulated_grads,
                    outer_grads
                )
            
            del outer_grads  # Free memory
        
        # Average gradients
        avg_grads = jax.tree_util.tree_map(
            lambda g: g / task_batch,
            accumulated_grads
        )
        del accumulated_grads
        
        # Update model
        updates, opt_state_outer = optim_outer.update(
            avg_grads, opt_state_outer, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        del avg_grads, updates
        
        # Log
        if epoch % 10 == 0:
            avg_inner = sum(inner_losses) / len(inner_losses)
            avg_outer = sum(outer_losses) / len(outer_losses)
            print(f"Epoch {epoch}: Inner Loss = {avg_inner:.4f}, Outer Loss = {avg_outer:.4f}")
        
        # Clear cache periodically
        if epoch % 5 == 0:
            import gc
            gc.collect()

    return model


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

    sampler = Sample_Task(dataset, key)

    sampler.sample()

    shape = dataset[0][0].shape

    model = CNN(key=key, channels= shape[0], width=shape[1], height=shape[2])

    model = train(model, sampler)


if __name__ == "__main__":
    main()
