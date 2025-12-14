import optax
import jax
import jax.numpy as jnp
import equinox as eqx
import tqdm as tqdm
from jaxtyping import Array, Float, Int, PyTree

from sampleTask import FCS_loader

from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy

from torchvision import datasets, transforms
from torch import manual_seed

from model.cnn import CNN

from functools import partial

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

SEED = 42
KEY = jax.random.PRNGKey(SEED)

manual_seed(seed=SEED)
class iMAML:
    def __init__(
            self,
            alpha: Float = .01,
            lambda_:  Float = 1.,
            regu_coef: Float = 1.,
            cg_damping: Float = 10.,
            cg_steps: Int = 5) -> None:
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regu_coef = regu_coef
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps

    def loss_acc(self, model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"], key) -> tuple[Float[Array, ""], Float]:
        pred_y = jax.vmap(model, in_axes=(0,None))(x, key)
        loss = cross_entropy(pred_y, y)
        acc = accuracy(pred_y, y)
        return jnp.mean(loss), acc
    
    def loss(self, model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"], key) -> Float[Array, ""]:
        pred_y = jax.vmap(model, in_axes=(0,None))(x, key)
        loss = cross_entropy(pred_y, y)
        return jnp.mean(loss)

    def inner_loop(
            self,
            model: CNN,
            support_set: tuple[Float[Array, "Channels Width Height"], Int],
            key
    ):
        
        gradient_descent = lambda p, p0, g: p - self.alpha * (g + self.lambda_ * (p - p0))

        init_params, _ = eqx.partition(model, eqx.is_array)
            
        gloss = eqx.filter_value_and_grad(self.loss_acc, has_aux=True)

        (loss_value, acc), grads = eqx.filter_vmap(gloss, in_axes=(None, 0,0, None))(model, support_set[0], support_set[1], key)

        avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

        params, _ = eqx.partition(model, eqx.is_array)

        params = jax.tree_util.tree_map(gradient_descent, params, init_params, avg_grads)

        return params, (loss_value, acc)
        
    
    def hessian_vector_product(
            self,
            params: PyTree,
            static,
            support_set: tuple[Float[Array, "Channels Width Height"], Int],
            key
    ):
        loss_fn = eqx.filter_vmap(eqx.filter_grad(self.loss), in_axes=(None, 0,0, None))
        train_loss = lambda x: loss_fn(eqx.combine(x, static), support_set[0], support_set[1], key)
        _, hvp_fn = jax.linearize(train_loss, params)
        mean = lambda g: jnp.mean(g, axis=0)

        def _hvp_damping(tangents):
            damping = lambda h, t: (1. + self.regu_coef) * mean(t) + mean(h) /(self.lambda_ + self.cg_damping)
            return jax.tree_util.tree_map(damping, hvp_fn(tangents), tangents)
        return _hvp_damping
    
    @eqx.filter_jit
    def task_gradient(self, model, support_set, query_set, key):

        subkey1, subkey2, subkey3, key = jax.random.split(key, 4)

        params, static = eqx.partition(model, eqx.is_array)

        addapted_params, (loss_value, inner_acc) = self.inner_loop(model, support_set, subkey1)

        model = eqx.nn.inference_mode(
            eqx.combine(addapted_params, static), 
            value=True
            )

        gloss = eqx.filter_value_and_grad(self.loss_acc, has_aux=True)
        (outer_loss, outer_acc), outer_grads = eqx.filter_vmap(gloss, in_axes=(None, 0,0, None))(model, query_set[0], query_set[1], subkey2)

        avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), outer_grads)

        hvp_fn = self.hessian_vector_product(
            addapted_params, static, support_set, subkey3
        )

        outer_grads, _ = jax.scipy.sparse.linalg.cg(
            hvp_fn,
            avg_grads,
            maxiter=self.cg_steps
        )

        return outer_grads, jnp.mean(loss_value).astype(float), jnp.mean(inner_acc).astype(float), jnp.mean(outer_loss).astype(float), jnp.mean(outer_acc).astype(float), key
    
    def train(
            self,
            model: CNN,
            sampler: FCS_loader,
            key,
            task_batch: Int = 5,
            inner_batch: Int = 5,
            epochs: Int = 100,

    ):
        optim_outer = optax.adamw(self.alpha / 10)
        opt_state_outer = optim_outer.init(eqx.filter(model, eqx.is_array))
        inner_losses = []
        outer_losses = []
        inner_acces = []
        outer_acces = []

        pbar = tqdm.tqdm(range(epochs))

        for epoch in pbar:


            support_set, query_set = sampler.sample_batch()
            model = eqx.nn.inference_mode(model, value=False)
            
            outer_grads, inner_loss, inner_acc, outer_loss, outer_acc, key = self.task_gradient(model, support_set, query_set, key)
            inner_losses.append(inner_loss.item())
            outer_losses.append(outer_loss.item())
            inner_acces.append(inner_acc.item())
            outer_acces.append(outer_acc.item())

            updates, opt_state_outer = optim_outer.update(outer_grads, opt_state_outer, eqx.filter(model, eqx.is_array))

            model = eqx.apply_updates(model, updates=updates)
            del updates, outer_grads

            if (epoch + 1) % 10 == 0:
                avg_inner = sum(inner_losses) / len(inner_losses)
                avg_outer = sum(outer_losses) / len(outer_losses)
                avg_inner_acc = sum(inner_acces) / len(inner_acces)
                avg_outer_acc = sum(outer_acces) / len(outer_acces)

                pbar.set_postfix({
                    "Iter" : f'{epoch + 1}',
                    "train loss" : f'{avg_inner:.4f}',
                    "train acc" : f'{avg_inner_acc:.4f}',
                    "val loss" : f'{avg_outer:.4f}',
                    "val acc" : f'{avg_outer_acc:.4f}'
                })
                # print(f"Epoch {epoch}: Inner Loss = {avg_inner:.4f}, Outer Loss = {avg_outer:.4f}")
                inner_losses = []
                outer_losses = []
                inner_acces = []
                outer_acces = []
        
            # Clear cache periodically
            jax.clear_caches()
        return model
                

def main():

    device = jax.devices('gpu')[0]
    jax.config.update('jax_default_device', device)


    normalize_data = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.RandomRotation((-180,180)),
        transforms.ColorJitter(),
        transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.Omniglot(
        "Omniglot",
        transform= normalize_data,
        download=True,
        background=True
    )

    key, subkey = jax.random.split(KEY)

    sampler = FCS_loader(dataset, subkey, batch_size=8)

    shape = dataset[0][0].shape

    model = CNN(key=key, channels= shape[0], width=shape[1], height=shape[2])

    maml = iMAML(alpha=1e-3)

    model = maml.train(model, sampler, subkey, inner_batch=5, epochs=5000)


if __name__ == "__main__":
    main()
