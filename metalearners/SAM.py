import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, PyTree, Bool

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

    @jax.jit
    def apply_pertibation(self,
                   params : PyTree,
                   grads : PyTree
                   ):
        
        grad_norm = self._grad_norm(params, grads)
        scale = self.rho / (grad_norm + 1e-12)
        
        def _epsilon(carry, _):
            param, grad = carry
            e_w = (jnp.pow(param, 2) if self.adaptive else 1.0) * grad * scale
            param = param + e_w
            return param
        
        p_params = jax.tree_util.tree_map(_epsilon, (params, grads))
        return params, p_params
    
    @jax.jit
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
        def _norm(carry, _):
            param, grad = carry
            return optax.tree.norm((jnp.abs(param) if self.adaptive else 1.0) * grad)
        
        norm = jax.tree_util.tree_map(_norm, (params, grads))

        return optax.tree.norm(norm)