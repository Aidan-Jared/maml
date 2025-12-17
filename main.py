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

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

SEED = 42
KEY = jax.random.PRNGKey(SEED)

manual_seed(seed=SEED)
class iMAML:
    def __init__(
            self,
            alpha: Float = .01,
            beta: Float = .01,
            grad_clip: Float = 1.,
            lambda_:  Float = 1.,
            inner_steps : Int = 5,
            cg_damping: Float = 10.,
            cg_steps: Int = 5) -> None:
        self.alpha = alpha
        self.beta = beta
        self.grad_clip = grad_clip
        self.lambda_ = lambda_
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps
        self.inner_steps = inner_steps

    def loss_acc(self, model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"], key: PRNGKeyArray | None = None) -> tuple[Float[Array, ""], Float]:
        pred_y = jax.vmap(model, in_axes=(0,None))(x, key)
        loss = cross_entropy(pred_y, y)
        acc = accuracy(pred_y, y)
        return jnp.mean(loss), acc
    
    def loss(self, model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"], key: PRNGKeyArray | None = None) -> Float[Array, ""]:
        pred_y = jax.vmap(model, in_axes=(0,None))(x, key)
        loss = cross_entropy(pred_y, y)
        return jnp.mean(loss)

    def inner_loop(
            self,
            model: CNN,
            support_set: tuple[Float[Array, "Channels Width Height"], Int],
    ):
        
        gradient_descent = lambda p, p0, g: p - self.alpha * (g + self.lambda_ * (p - p0))

        params, static = eqx.partition(model, eqx.is_array)
        init_params = params
            
        losses = []
        acces = []

        for _ in range(self.inner_steps):

            (loss_value, acc), grads = eqx.filter_value_and_grad(self.loss_acc, has_aux=True)(eqx.combine(params, static), support_set[0], support_set[1])
            # avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
            losses.append(loss_value)
            acces.append(acc)

            # grad_norm = optax.tree_utils.tree_norm(avg_grads)
            # jax.debug.print("inner_grad: {}", grad_norm)

            params = jax.tree_util.tree_map(gradient_descent, params, init_params, grads)
            

        return params, (jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(acces)))
        
    
    def hessian_vector_product(
            self,
            params: PyTree,
            static,
            support_set: tuple[Float[Array, "Channels Width Height"], Int],
    ):
        
        # static = eqx.nn.inference_mode(static, value=True)
        
        def _s_loss(p):
            losses = self.loss(eqx.combine(p, static), support_set[0], support_set[1])
            return jnp.mean(losses)
        
        grad_fn = jax.grad(_s_loss)
        _, hvp_fn = jax.linearize(grad_fn, params)

        def _hvp_damping(tangents):
            Hv = hvp_fn(tangents)
            return jax.tree_util.tree_map(lambda h, t: h + self.lambda_ * t,
                                          Hv, tangents)
        return _hvp_damping
    
    def solve_single_task(self, model, support, query):

        params, static = eqx.partition(model, eqx.is_array)

        adapted_params, (inner_loss, inner_acc) = self.inner_loop(model, support)

        d_static = eqx.nn.inference_mode(static, value=True)

        a_model = eqx.combine(adapted_params, d_static)

        (outer_loss, outer_acc), outer_grads = eqx.filter_value_and_grad(
            self.loss_acc, has_aux=True
        )(a_model, query[0], query[1])

        hvp_fn = self.hessian_vector_product(
            adapted_params, static, support
        )

        v, _ = jax.scipy.sparse.linalg.cg(
            hvp_fn,
            outer_grads,
            maxiter=self.cg_steps
        )

        # v = outer_grads

        v = jax.tree.map(
            lambda p, p_star, v_i: self.lambda_ * (p - p_star + v_i),
            params,
            adapted_params,
            v
        )

        return v, inner_loss, inner_acc, outer_loss, outer_acc
    
    @eqx.filter_jit
    def task_gradient(self, model, support_set, query_set, key):

        vs, inner_losses, inner_acces, outer_losses, outer_acces = eqx.filter_vmap(self.solve_single_task, in_axes=(None, 0,0))(model, support_set, query_set)
        outer_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vs)

#         residual = hvp_fn(outer_grads)
#         residual_norm = optax.tree_utils.tree_norm(
#             jax.tree_util.tree_map(lambda r, b: r - b, residual, avg_grads)
# )

#         jax.debug.print("residual norm: {}", residual_norm)

        return outer_grads, jnp.mean(inner_losses).astype(float), jnp.mean(inner_acces).astype(float), jnp.mean(outer_losses).astype(float), jnp.mean(outer_acces).astype(float), key
    
    def train(
            self,
            model: CNN,
            sampler: FCS_loader,
            key,
            epochs: Int = 100,

    ):
        optim_outer = optax.chain(optax.clip_by_global_norm(self.grad_clip), optax.sgd(self.beta))
        opt_state_outer = optim_outer.init(eqx.filter(model, eqx.is_array))
        inner_losses = []
        outer_losses = []
        inner_acces = []
        outer_acces = []

        pbar = tqdm.tqdm(range(epochs))

        for epoch in pbar:


            support_set, query_set = sampler.sample_batch("cpu")
            
            outer_grads, inner_loss, inner_acc, outer_loss, outer_acc, key = self.task_gradient(model, support_set, query_set, key)
            inner_losses.append(inner_loss.item())
            outer_losses.append(outer_loss.item())
            inner_acces.append(inner_acc.item())
            outer_acces.append(outer_acc.item())

            # grad_norm = optax.tree_utils.tree_norm(outer_grads)
            # jax.debug.print("grads: {}", grad_norm)

            updates, opt_state_outer = optim_outer.update(outer_grads, opt_state_outer, eqx.filter(model, eqx.is_array))

            model = eqx.apply_updates(model, updates=updates)
            del updates, outer_grads

            if (epoch + 1) % 5 == 0:
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

            jax.clear_caches()
        return model
                

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

    maml = iMAML(alpha=1e-4, beta=1e-4, lambda_=1., cg_steps=10, cg_damping=1., grad_clip=5., inner_steps=50)

    model = eqx.nn.inference_mode(model, value=True)

    model = maml.train(model, sampler, subkey, epochs=5000)


if __name__ == "__main__":
    main()
