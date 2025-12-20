import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, PyTree, Array, Int, PRNGKeyArray
from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy
from SAM import SAM

import equinox as eqx

from sampleTask import FCS_loader

import tqdm as tqdm
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'

class ANIL:
    def __init__(
            self,
            alpha: Float = 1e-3,
            beta: Float = 1e-4,
            inner_step: Int = 5,
            task_batch: Int = 5
            ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.inner_step = inner_step
        self.task_batch = task_batch
    
    def loss(
            self,
            model: PyTree,
            x: Float[Array, " batch C H W"], 
            y: Int[Array, " batch"],
            key: PRNGKeyArray | None
    ) -> tuple[Float, Float]:
        pred_y = jax.vmap(model, in_axes=(0,0))(x, key)
        loss = cross_entropy(pred_y, y)
        acc = accuracy(pred_y, y)
        return jnp.mean(loss), acc
    
    def inner_loop(
            self,
            backbone: PyTree,
            head: PyTree,
            support: tuple[Array, Array],
            optim: optax.GradientTransformationExtraArgs,
            opt_state: PyTree,
            key: PRNGKeyArray
    ) -> tuple[tuple[PyTree, PyTree, PRNGKeyArray], tuple[Array, Array]]:
        x, y = support
        key, *subkey = jax.random.split(key, x.shape[0]+1)
        subkeys = jnp.stack(subkey)
        x, _ = jax.vmap(backbone, in_axes=(0, 0))(x, subkeys)
        

        params, static = eqx.partition(head, eqx.is_array)
        
        @eqx.filter_checkpoint
        def make_step(
                carry,
                _
        ):
            params, opt_state, key = carry
            key, *subkey = jax.random.split(key, x.shape[0]+1)
            subkeys = jnp.stack(subkey)
            head = eqx.combine(params, static)

            (loss_value, acc), grads = eqx.filter_value_and_grad(self.loss, has_aux=True)(head, x, y, subkeys)

            update, opt_state = optim.update(grads, opt_state, params)
            head = eqx.apply_updates(head, update)

            params,  _ = eqx.partition(head, eqx.is_array)

            return (params, opt_state, key), (loss_value,  acc)
        
        return jax.lax.scan(
            make_step,
            (params, opt_state, key),
            None,
            length=self.inner_step
        )
    
    @eqx.filter_jit
    def outer_loop(
            self,
            model: PyTree,
            support: tuple[Array, Array],
            query: tuple[Array, Array],
            key: PRNGKeyArray
    ):
        backbone = model.backbone
        head = model.head
        _, static = eqx.partition(head, eqx.is_array)

        optim = optax.sgd(self.beta)
        opt_state = optim.init(eqx.filter(head, eqx.is_array))

        (params, _, key), (support_loss,  support_acc) = self.inner_loop(
            backbone, 
            head, 
            support,
            optim,
            opt_state,
            key
            )

        up_head = eqx.combine(params, static)

        model = eqx.tree_at(lambda m: m.head, model, up_head)
        del up_head, head, backbone, static

        (query_loss, query_acc), outer_grads = eqx.filter_value_and_grad(self.loss, has_aux=True)(model, query[0], query[1], None)

        return jnp.mean(support_loss).astype(float),  jnp.mean(support_acc).astype(float), jnp.mean(query_loss).astype(float), jnp.mean(query_acc).astype(float), outer_grads
    

    def train(
            self,
            model: PyTree,
            sampler: FCS_loader,
            key: PRNGKeyArray,
            epochs: Int = 100,
    ):
        optim_outer = optax.sgd(self.alpha)

        opt_state_outer = optim_outer.init(eqx.filter(model, eqx.is_array))
        pbar = tqdm.tqdm(range(epochs))
        for epoch in pbar:
            inner_losses=[]
            outer_losses=[]
            inner_acces=[]
            outer_acces=[]
            accumulated_grads = None

            for _ in range(self.task_batch):
                support, query = sampler.sample_batch()
                key, *subkey = jax.random.split(key, support[0].shape[0] + 1)
                subkeys = jnp.stack(subkey)
                support_loss, support_acc, query_loss, query_acc, outer_grads = jax.vmap(self.outer_loop, in_axes=(None, 0, 0, 0))(model, support, query, subkeys)

                inner_losses.extend(support_loss.tolist())
                outer_losses.extend(query_loss.tolist())
                inner_acces.extend(support_acc.tolist())
                outer_acces.extend(query_acc.tolist())
                
                avg_grads = jax.tree.map(
                    lambda g: jnp.mean(g, axis=0) if isinstance(g, jnp.ndarray) else g,
                    outer_grads
                )
                if accumulated_grads is None:
                    accumulated_grads = avg_grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda acc, new: acc + new,
                        accumulated_grads,
                        avg_grads
                    )

                del support_loss, query_loss, support_acc, query_acc, outer_grads, avg_grads

            avg_grads = jax.tree_util.tree_map(
                lambda g: g / self.task_batch,
                accumulated_grads
            )
            updates, opt_state_outer = optim_outer.update(avg_grads, opt_state_outer, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)

            if epoch % 5 == 0:
                avg_inner_loss = sum(inner_losses) / len(inner_losses)
                avg_outer_loss = sum(outer_losses) / len(outer_losses)
                avg_inner_acc = sum(inner_acces) / len(inner_acces)
                avg_outer_acc = sum(outer_acces) / len(outer_acces)

                pbar.set_postfix({
                    "Iter" : f'{epoch + 1}',
                    "train loss" : f'{avg_inner_loss:.4f}',
                    "train acc" : f'{avg_inner_acc:.4f}',
                    "val loss" : f'{avg_outer_loss:.4f}',
                    "val acc" : f'{avg_outer_acc:.4f}'
                })
                jax.clear_caches()
        
        return model
    

class SAM_ANIL(ANIL):
    def __init__(
            self, 
            alpha: Float = 1e-3, 
            beta: Float = 1e-4, 
            inner_step: Float = 5, 
            task_batch: Float = 5
            ) -> None:
        super().__init__(alpha, beta, inner_step, task_batch)

    @eqx.filter_jit
    def outer_loop(
            self,
            model: PyTree,
            support: tuple[Array, Array],
            query: tuple[Array, Array],
            sam: SAM,
            key: PRNGKeyArray
    ):
        backbone = model.backbone
        head = model.head
        _, static = eqx.partition(head, eqx.is_array)

        optim = optax.sgd(self.beta)
        opt_state = optim.init(eqx.filter(head, eqx.is_array))

        (params, _, key), (support_loss,  support_acc) = self.inner_loop(
            backbone, 
            head, 
            support,
            optim,
            opt_state,
            key
            )

        up_head = eqx.combine(params, static)

        model = eqx.tree_at(lambda m: m.head, model, up_head)
        del up_head, head, backbone, static

        (_, _), outer_grads = eqx.filter_value_and_grad(self.loss, has_aux=True)(model, query[0], query[1], None)

        params, static = eqx.partition(model, eqx.is_array)
        p_params = sam.apply_pertibation(params, outer_grads)
        p_model = eqx.combine(p_params, static)
        (query_loss, query_acc), outer_grads = eqx.filter_value_and_grad(self.loss, has_aux=True)(p_model, query[0], query[1], None)
        

        return jnp.mean(support_loss).astype(float),  jnp.mean(support_acc).astype(float), jnp.mean(query_loss).astype(float), jnp.mean(query_acc).astype(float), outer_grads
    
    def train(
        self,
        model: PyTree,
        sampler: FCS_loader,
        key: PRNGKeyArray,
        rho: Float = .05,
        epochs: Int = 100,
    ):
        optim = optax.sgd(self.beta)
        sam = SAM(optim, rho=rho)

        opt_state_outer = optim.init(eqx.filter(model, eqx.is_array))
        pbar = tqdm.tqdm(range(epochs))
        for epoch in pbar:
            inner_losses=[]
            outer_losses=[]
            inner_acces=[]
            outer_acces=[]
            accumulated_grads = None

            for _ in range(self.task_batch):
                support, query = sampler.sample_batch()
                key, *subkey = jax.random.split(key, support[0].shape[0] + 1)
                subkeys = jnp.stack(subkey)
                support_loss, support_acc, query_loss, query_acc, outer_grads = jax.vmap(self.outer_loop, in_axes=(None, 0, 0, None, 0))(model, support, query, sam, subkeys)

                inner_losses.extend(support_loss.tolist())
                outer_losses.extend(query_loss.tolist())
                inner_acces.extend(support_acc.tolist())
                outer_acces.extend(query_acc.tolist())
                
                avg_grads = jax.tree.map(
                    lambda g: jnp.mean(g, axis=0) if isinstance(g, jnp.ndarray) else g,
                    outer_grads
                )
                if accumulated_grads is None:
                    accumulated_grads = avg_grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda acc, new: acc + new,
                        accumulated_grads,
                        avg_grads
                    )

                del support_loss, query_loss, support_acc, query_acc, outer_grads, avg_grads

            avg_grads = jax.tree_util.tree_map(
                lambda g: g / self.task_batch,
                accumulated_grads
            )

            updates, opt_state_outer = sam.update(avg_grads, opt_state_outer, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)

            if epoch % 5 == 0:
                avg_inner_loss = sum(inner_losses) / len(inner_losses)
                avg_outer_loss = sum(outer_losses) / len(outer_losses)
                avg_inner_acc = sum(inner_acces) / len(inner_acces)
                avg_outer_acc = sum(outer_acces) / len(outer_acces)

                pbar.set_postfix({
                    "Iter" : f'{epoch + 1}',
                    "train loss" : f'{avg_inner_loss:.4f}',
                    "train acc" : f'{avg_inner_acc:.4f}',
                    "val loss" : f'{avg_outer_loss:.4f}',
                    "val acc" : f'{avg_outer_acc:.4f}'
                })
                jax.clear_caches()
        
        return model
    