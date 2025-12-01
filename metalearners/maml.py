import equinox as eqx
import jax.numpy as jnp
from jax import grad, tree_util, lax

from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy

from metalearners.base import MetaLearner

class MAML(MetaLearner):
    def __init__(self, model, num_steps=5, alpha =.1):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.alpha = alpha

    def loss(self, params, state, inputs, targets, args):
        logits, state = self.model.apply(params, state, inputs, *args)
        loss = jnp.mean(cross_entropy(logits, targets))
        logs = {
            "loss": loss,
            "accuracy": accuracy(logits, targets)
        }
        return loss, (state, logs)
    
    def adapt(self, init_params, state, inputs, targets, args):
        loss_grad = eqx.filter_grad(self.loss, has_aux=True)

        gradient_descent = lambda p, g: p - self.alpha * g

        def _gradient_update(params,_):
            grads, (_, logs) = loss_grad(params, state, inputs, targets, args)
            params = tree_util.tree_map(gradient_descent, params, grads)

            return params, logs

        return lax.scan(
            _gradient_update,
            init_params,
            None,
            length=self.num_steps
        )
    
    def meta_init(self, key, *args, **kwargs):
        return self.model.init(key, *args, **kwargs)