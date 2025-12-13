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
            param = param + e_w
            return param
        
        p_params = jax.tree_util.tree_map(_epsilon, params, grads)
        return params, p_params
    
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

def loss( model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"]) -> tuple[Float[Array, ""], Float]:
    pred_y = jax.vmap(model)(x)
    loss = jnp.mean(cross_entropy(pred_y, y))
    acc = accuracy(pred_y, y)
    return (loss, acc)

def train(
        model: CNN,
        sampler: FCS_loader,
        optim: optax.GradientTransformationExtraArgs,
        iterations: Int = 100,
        rho: Float = .01
) -> CNN:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    sam = SAM(optim, rho)

    @eqx.filter_jit
    def step(model: CNN, X: Float[Array, " batch 1 28 28"], y: Float[Array, " batch"], opt_state: PyTree):
        (loss_value, acc), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, X, y)

        params, static = eqx.partition(model, eqx.is_array)
        params, p_params = sam.apply_pertibation(params, grads)

        p_model = eqx.combine(p_params, static)
        (p_loss_value, p_acc), grads = eqx.filter_value_and_grad(loss, has_aux=True)(p_model, X, y)

        updates, opt_state = sam.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, acc, p_loss_value, p_acc



    train_losses = []
    train_acces = []
    p_train_losses = []
    p_train_acces = []

    test_losses = []
    test_acces = []

    pbar = tqdm.tqdm(range(iterations))
    for iter in pbar:
    
        support_set, query_set = sampler.sample_batch()
        support_x = support_set[0].reshape(-1, *support_set[0].shape[2:])
        support_y = support_set[1].reshape(-1)

        query_x = query_set[0].reshape(-1, *query_set[0].shape[2:])
        query_y = query_set[1].reshape(-1)


        model, opt_state, loss_value, acc, p_loss_value, p_acc = step(model, support_x, support_y, opt_state)
        train_losses.append(loss_value.item())
        p_train_losses.append(p_loss_value.item())
        train_acces.append(acc.item())
        p_train_acces.append(p_acc.item())
    
        val_loss, val_acc = eqx.filter_jit(loss)(model, query_x, query_y)
        test_losses.append(val_loss.item())
        test_acces.append(val_acc.item())

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
    n_way = 5
    normalize_data = transforms.Compose(
        [
        transforms.ToTensor(),
        # transforms.Resize(28),
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

    sampler = FCS_loader(dataset, key, batch_size=16, n_ways=5, k_shot=1, q_query=5)

    shape = dataset[0][0].shape
    model = CNN(key=subkey, channels= shape[0], width=shape[1], height=shape[2], n_way = n_way)

    lr = 1e-3
    optim = optax.sgd(learning_rate=lr)

    model = train(model, sampler, optim)
    
if __name__ == "__main__":
    main()