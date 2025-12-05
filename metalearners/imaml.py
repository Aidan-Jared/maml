import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, PyTree
import equinox as eqx
import tqdm as tqdm
import optax
  
from functools import partial

from jax_meta.utils.losses import cross_entropy

from ..model.cnn import CNN

from ..sampleTask import Sample_Task



class iMAML:
    def __init__(
            self,
            alpha: Float = .1,
            lambda_:  Float = 1.,
            regu_coef: Float = 1.,
            cg_damping: Float = 10.,
            cg_steps: Int = 5) -> None:
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regu_coef = regu_coef
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps

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
            batch: Int = 5
    ):
        
        gradient_descent = lambda p, p0, g: p - self.alpha * (g + self.lambda_ * (p - p0))

        init_params, static = eqx.partition(model, eqx.is_array)

        def make_step(
                params,
                _
        ):
            model = eqx.combine(params, static)

            loss_value, grads = eqx.filter_value_and_grad(self.loss)(model, support_set[0], support_set[1])

            params = jax.tree_util.tree_map(gradient_descent, params, init_params, grads)

            return params, loss_value
        
        return jax.lax.scan(
            make_step,
            init_params,
            None,
            length= batch
        )
    
    def hessian_vector_product(
            self,
            params: PyTree,
            static,
            support_set: tuple[Float[Array, "Channels Width Height"], Int],
    ):
        loss_fn = eqx.filter_grad(self.loss)
        train_loss = lambda x: loss_fn(eqx.combine(x, static), support_set[0], support_set[1])
        _, hvp_fn = jax.linearize(train_loss, params)

        def _hvp_damping(tangents):
            damping = lambda h, t: (1. + self.regu_coef) * t + h /(self.lambda_ + self.cg_damping)
            return jax.tree_util.tree_map(damping, hvp_fn(tangents), tangents)
        return _hvp_damping
    
    @eqx.filter_jit
    def task_gradient(self, model, inner_batch, support_set, query_set):

        params, static = eqx.partition(model, eqx.is_array)
        del params

        addapted_params, loss_value = self.inner_loop(model, support_set, inner_batch)


        outer_loss, outer_grads = eqx.filter_value_and_grad(self.loss)(eqx.combine(addapted_params, static), query_set[0], query_set[1])

        hvp_fn = self.hessian_vector_product(
            addapted_params, static, support_set
        )

        outer_grads, _ = jax.scipy.sparse.linalg.cg(
            hvp_fn,
            outer_grads,
            maxiter=self.cg_steps
        )

        del addapted_params
        return outer_grads, jnp.mean(loss_value).astype(float), outer_loss.astype(float)
    
    def train(
            self,
            model: CNN,
            sampler: Sample_Task,
            task_batch: Int = 5,
            inner_batch: Int = 5,
            epochs: Int = 100,

    ):
        optim_outer = optax.adamw(self.alpha / 10)
        opt_state_outer = optim_outer.init(eqx.filter(model, eqx.is_array))

        for epoch in tqdm.tqdm(range(epochs)):
            inner_losses = []
            outer_losses = []

            accumulated_grads = []

            for _ in range(task_batch):
                support_set, query_set = sampler.sample()
                
                outer_grads, inner_loss, outer_loss = self.task_gradient(model, inner_batch, support_set, query_set)
                inner_losses.append(inner_loss)
                outer_losses.append(outer_loss)
                outer_grads = jax.lax.stop_gradient(outer_grads)

                accumulated_grads.append(outer_grads)
                
                del outer_grads

            avg_grads = jax.tree_util.tree_map(
                lambda *grads: jnp.mean(jnp.stack(grads), axis=0),
                *accumulated_grads
            )

            del accumulated_grads

            updates, opt_state_outer = optim_outer.update(avg_grads, opt_state_outer, eqx.filter(model, eqx.is_array))

            model = eqx.apply_updates(model, updates=updates)
            del updates, avg_grads

            if epoch % 10 == 0:
                avg_inner = sum(inner_losses) / len(inner_losses)
                avg_outer = sum(outer_losses) / len(outer_losses)
                print(f"Epoch {epoch}: Inner Loss = {avg_inner:.4f}, Outer Loss = {avg_outer:.4f}")
        
            # Clear cache periodically
            jax.clear_caches()
        return model
   