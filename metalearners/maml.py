import equinox as eqx
import jax.numpy as jnp
import jax
from ..model.cnn import CNN

import optax
import tqdm as tqdm

from jax_meta.utils.losses import cross_entropy
from jaxtyping import Array, Float, Int, PyTree

from functools import partial

from sampleTask import Sample_Task

class MAML:
    def __init__(
            self
            ) -> None:
        pass

    @eqx.filter_jit
    def loss(self, model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"]) -> Float[Array, ""]:
        pred_y = jax.vmap(model)(x)
        loss = cross_entropy(pred_y, y)
        
        return jnp.mean(loss)

    @partial(eqx.filter_jit, donate="none")
    def inner_loop(
            self,
            model: CNN,
            support_set: tuple[Float[Array, "Channels Width Height"], Int],
            query_set: tuple[Float[Array, "Channels Width Height"], Int],
            alpha: Float,
            batch: Int = 5
    ):
        

        params, static = eqx.partition(model, eqx.is_array)
        optim = optax.sgd(alpha)
        opt_state = optim.init(params)
        
        def make_step(
            params, opt_state
        ):
            model = eqx.combine(params, static)

            loss_value, grads = eqx.filter_value_and_grad(self.loss)(model, support_set[0], support_set[1])
            updates, opt_state = optim.update(
                grads, opt_state, params
            )

            params = eqx.apply_updates(params, updates)
            return (params, opt_state), loss_value
        
        (adapted_params, _), inner_losses = jax.lax.scan(
            make_step,
            (params, opt_state),
            None,
            length=batch
        )

        model = eqx.combine(adapted_params, static)

        outer_loss, outer_grads = eqx.filter_value_and_grad(self.loss)(model, query_set[0], query_set[1])

        del model, opt_state

        return jnp.mean(jnp.array(inner_losses)), outer_loss, outer_grads

    def train(
            self,
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
                inner_loss, outer_loss, outer_grads = self.inner_loop(
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

