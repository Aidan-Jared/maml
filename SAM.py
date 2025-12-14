import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, PyTree, Bool, Array, Int
from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy

from torchvision import datasets, transforms
from torch import manual_seed
import equinox as eqx

from sampleTask import FCS_loader
from model.cnn import CNN

from functools import partial
import tqdm as tqdm

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


SEED = 42
KEY = jax.random.PRNGKey(SEED)
manual_seed(SEED)

class SAM:
    def __init__(
            self, 
            base_optimizer : optax.GradientTransformationExtraArgs,
            rho : Float = .05,
            adaptive: Bool = False,
            ) -> None:
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive

    eqx.filter_jit
    def apply_pertibation(self,
                   params : PyTree,
                   grads : PyTree
                   ):
        
        grad_norm = self._grad_norm(params, grads)
        scale = self.rho / (grad_norm + 1e-12)
        
        def _epsilon(param, grad):
            
            e_w = (jnp.pow(param, 2) if self.adaptive else 1.0) * grad * scale
            return param + e_w
        
        p_params = jax.tree_util.tree_map(_epsilon, params, grads)
        return p_params
    
    eqx.filter_jit
    def update(
            self,
            grads: PyTree,
            opt_state: optax.OptState,
            params: PyTree,
    ):
        return self.base_optimizer.update(grads, opt_state, params)
    

    def _grad_norm(
            self,
            params : PyTree,
            grads : PyTree
    ):
        def _norm(param, grad):
            return optax.tree.norm((jnp.abs(param) if self.adaptive else 1.0) * grad)
        
        norm = jax.tree_util.tree_map(_norm, params, grads)

        return optax.tree.norm(norm)

def loss( model: CNN, x: Float[Array, " task 1 28 28"], y: Int[Array, " task"], key) -> tuple[Float[Array, ""], Float]:
    pred_y = jax.vmap(model, in_axes=(0, None))(x, key)
    loss = jnp.mean(cross_entropy(pred_y, y))
    acc = accuracy(pred_y, y)
    return (loss, acc)

def train(
        model: CNN,
        sampler: FCS_loader,
        optim: optax.GradientTransformationExtraArgs,
        key,
        iterations: Int = 100,
        rho: Float = .01
) -> CNN:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    sam = SAM(optim, rho)

    @eqx.filter_jit
    def step(model: CNN, X: Float[Array, " batch task 1 28 28"], y: Float[Array, " batch task"], opt_state: PyTree, key):
        gloss = eqx.filter_value_and_grad(loss, has_aux=True)
        (loss_value, acc), grads = eqx.filter_vmap(gloss,in_axes=(None, 0, 0, None))(model, X, y, key)

        avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
        params, static = eqx.partition(model, eqx.is_array)
        p_params = sam.apply_pertibation(params, avg_grads)

        p_model = eqx.combine(p_params, static)
        (p_loss_value, p_acc), grads = eqx.filter_vmap(gloss,in_axes=(None, 0, 0, None))(p_model, X, y, key)

        avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

        updates, opt_state = sam.update(avg_grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, jnp.mean(loss_value), jnp.mean(acc), jnp.mean(p_loss_value), jnp.mean(p_acc)



    train_losses = []
    train_acces = []
    p_train_losses = []
    p_train_acces = []

    test_losses = []
    test_acces = []

    pbar = tqdm.tqdm(range(iterations))
    vloss = eqx.filter_vmap(loss, in_axes=(None, 0,0, None))
    for iter in pbar:
    
        support_set, query_set = sampler.sample_batch()
        support_x = support_set[0]#.reshape(-1, *support_set[0].shape[2:])
        support_y = support_set[1]#.reshape(-1)

        query_x = query_set[0]#.reshape(-1, *query_set[0].shape[2:])
        query_y = query_set[1]#.reshape(-1)
        model = eqx.nn.inference_mode(model, value=False)
        model, opt_state, loss_value, acc, p_loss_value, p_acc = step(model, support_x, support_y, opt_state, key)
        train_losses.append(loss_value.item())
        p_train_losses.append(p_loss_value.item())
        train_acces.append(acc.item())
        p_train_acces.append(p_acc.item())

        model = eqx.nn.inference_mode(model, value=True)
        val_loss, val_acc = eqx.filter_jit(vloss)(model, query_x, query_y, key)
        test_losses.append(jnp.mean(val_loss).item())
        test_acces.append(jnp.mean(val_acc).item())

        if (iter + 1) % 10 == 0:
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_acc = sum(train_acces) / len(train_acces)
            avg_p_train_loss = sum(p_train_losses) / len(p_train_losses)
            avg_p_train_acc = sum(p_train_acces) / len(p_train_acces)
            avg_val_loss = sum(test_losses) / len(test_losses)
            avg_val_acc = sum(test_acces) / len(test_acces)

            pbar.set_postfix({
                "Iter" : f'{iter + 1}',
                "train loss" : f'{avg_train_loss:.4f}',
                "train acc" : f'{avg_train_acc:.4f}',
                "p_train loss" : f'{avg_p_train_loss:.4f}',
                "p_train acc" : f'{avg_p_train_acc:.4f}',
                "val loss" : f'{avg_val_loss:.4f}',
                "val acc" : f'{avg_val_acc:.4f}'
            })

            # print(f"Iter {iter + 1}: train loss = {avg_train_loss:.4f}, train acc = {avg_train_acc:.4f}")
            # print(f"p_train loss = {avg_p_train_loss:.4f}, p_train acc = {avg_p_train_acc:.4f}")
            # print(f"val loss = {avg_val_loss:.4f}, val acc = {avg_val_acc:.4f}")
            train_losses = []
            train_acces = []
            p_train_losses = []
            p_train_acces = []

            test_losses = []
            test_acces = []

    return model
    
def main():

    device = jax.devices('gpu')[0]
    jax.config.update('jax_default_device', device)
    n_way = 5
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

    key, subkey1, subkey2 = jax.random.split(KEY, 3)

    sampler = FCS_loader(dataset, key, batch_size=8, n_ways=5, k_shot=1, q_query=15)

    shape = dataset[0][0].shape
    model = CNN(key=subkey1, channels= shape[0], width=shape[1], height=shape[2], n_way = n_way)

    lr = 3e-2
    optim = optax.sgd(learning_rate=lr)

    model = train(model, sampler, optim, subkey2, iterations=int(5e4))
    
if __name__ == "__main__":
    main()