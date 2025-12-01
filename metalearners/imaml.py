import jax.numpy as jnp
import optax
import jax

import equinox as eqx
from functools import partial

from metalearners.base import MetaLearnerState
from metalearners.maml import MAML

class iMAMAL(MAML):
    def __init__(self, model, num_steps=5, alpha = .1, lambda_=1., regu_coef=1., cg_damping=10., cg_steps=5):
        super().__init__(model, num_steps=num_steps, alpha=alpha)
        self.lambda_ = lambda_
        self.regu_ceof = regu_coef
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps

    def adapt(self, init_params, state, inputs, targets, args):
        loss_grad = eqx.filter_grad(self.loss, has_aux=True)

        gradient_descent = lambda p, p0, g: p - self.alpha * (g + self.lambda_ * (p - p0))

        def _gradient_update(params, _):

            grads, (_, logs) = loss_grad(params, state, inputs, targets, args)
            params = jax.tree_util.tree_map(gradient_descent, params, init_params, grads)

            return params, logs

        return jax.lax.scan(
            _gradient_update,
            init_params,
            None, length=self.num_steps
        )
    
    def hessian_vector_product(self, params, state, inputs, targets, args):

        train_loss = lambda primals: self.loss(primals, state, inputs, targets, args)

        _, hvp_fn = jax.linearize(eqx.filter_grad(train_loss),  params)

        def _hvp_damping(tangets):
            damping = lambda h, t: (1. + self.regu_ceof) * t + h /(self.lambda_ + self.cg_damping)
            return jax.tree_util.tree_map(damping, hvp_fn(tangets), tangets)
        
        return _hvp_damping
    
    def grad_outer_loss(self, params, state, train, test, args):
        
        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def _grad_outer_loss(params, state, train, test):
            adapted_params, inner_logs = self.adapt(
                params, state, train.inputs, train.targets, args
            )

            grads, (state, outer_logs) = eqx.filter_grad(self.loss, has_aux=True)(
                adapted_params, state, test.inputs, test.targets, args
            )

            hvp_fn = self.hessian_vector_product(
                adapted_params, state, train.inputs, train.targets, args
            )

            outer_grads, _ = jax.scipy.sparse.linalg.cg(hvp_fn, grads, maxiter=self.cg_steps)

            return outer_grads, inner_logs, outer_logs, state
        
        outer_grads, inner_logs, outer_logs, states = _grad_outer_loss(
            params, state, train, test
        )

        outer_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), states)

        logs = {
            **{f'inner/{k}': v for (k, v) in inner_logs.items()},
            **{f'outer/{k}': v for (k, v) in outer_logs.items()}
        }
        return outer_grads, (state, logs)
    
    @partial(eqx.filter_jit, static_argnums=(0,5))
    def train_step(self, params, state, train, test, args):
        grads, (model_state, logs) = self.grad_outer_loss(
            params, state.model, train, test, args
        )

        updates, opt_state = self.optimizer.update(grads, state.optimizer, params)

        params = eqx.apply_updates(params, updates=updates)

        state = MetaLearnerState(model=model_state, optimizer=opt_state,  key=state.key)

        return params, state, logs