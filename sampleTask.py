import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from jaxtyping import PRNGKeyArray, Int, Array


# add batching (groups of tasks, 1 batch is 1 task)
class FCS_loader:
    def __init__(
            self,
            dataset: Dataset, 
            key: PRNGKeyArray, 
            batch_size: Int,
            n_ways: Int = 5, 
            k_shot: Int = 1, 
            q_query: Int = 15,
            device: str = "cpu"
            ) -> None:
        self.key=key
        self.n_ways=n_ways
        self.k_shot=k_shot
        self.q_query=q_query

        self.samples_per_class = self.k_shot + self.q_query

        self.batch_size = batch_size

        class_to_indices = {}
        all_data = []

        for idx, (data, label) in enumerate(dataset):
            if isinstance(data, jnp.ndarray):
                all_data.append(np.array(data))
            else:
                all_data.append(data.numpy())
            label_int = int(label)
            if label_int not in class_to_indices:
                class_to_indices[label_int] = []
            class_to_indices[label_int].append(idx)
        
        device = jax.devices(device)[0]

        all_data_np = np.stack(all_data)
        # if all_data_np.ndim == 4 and all_data_np.shape[1] == 1:
        #     all_data_np = all_data_np.squeeze(1)

        self.all_data = jax.device_put(all_data_np, device)
        

        self.num_classes = len(class_to_indices)
        max_samples_per_class = max(len(v) for v in class_to_indices.values())

        self.class_indicies = jax.device_put(
            jnp.full((self.num_classes, max_samples_per_class), -1, dtype=jnp.int32), device
        )

        self.class_lenghts = jax.device_put(
            jnp.zeros(self.num_classes, dtype=jnp.int32), device
        )

        for class_idx, (label, idx) in enumerate(sorted(class_to_indices.items())):
            num_samples = len(idx)
            self.class_indicies = self.class_indicies.at[class_idx, :num_samples].set(
                jnp.array(idx, dtype=jnp.int32)
            )

            self.class_lenghts = self.class_lenghts.at[class_idx].set(num_samples)
        

        self._sample_task_jit = jax.jit(
            self._sample_task_fn,
            static_argnames=["n_ways", "samples_per_class"]
        )

        self._sample_batch_jit = jax.jit(
            jax.vmap(self._sample_task_fn, in_axes=(0, None, None, None, None, None)),
            static_argnames=["n_ways", "samples_per_class"]
        )

    @staticmethod
    def _sample_task_fn(
        key: PRNGKeyArray,
        class_indicies: Array,
        class_lenghts: Array,
        all_data: Array,
        n_ways: Int,
        samples_per_class: Int
    ) -> tuple[Array, Array]:
        
        num_classes = class_indicies.shape[0]
        key, subkey = jax.random.split(key)
        shuffled_classes = jax.random.permutation(subkey, num_classes)
        selected_classes = jax.lax.dynamic_slice(shuffled_classes, (0,),(n_ways,))
        
        def sample_class(
                carry: tuple[PRNGKeyArray, Int], 
                class_idx: Int
                ) -> tuple[tuple[PRNGKeyArray, Int], tuple[Array, Array]]:
            key, new_label = carry
            key, subkey = jax.random.split(key)

            class_row = class_indicies[class_idx]
            mask = class_row > 0
            valid_idx = class_row[jnp.where(mask, class_row, 0)]
            # valid_idx = class_indicies[class_idx, :class_len]

            selected_idx = jax.random.choice(subkey, valid_idx, shape=(samples_per_class,), replace=False)

            data = all_data[selected_idx]

            labels = jnp.full(samples_per_class, new_label, dtype=jnp.int32)

            return (key, new_label + 1), (data, labels)
        
        (key, _), (all_class_data, all_class_labels) = jax.lax.scan(
            sample_class,
            (key, 0),
            selected_classes
        )

        all_class_data = all_class_data.reshape(-1, *all_class_data.shape[2:])
        all_class_labels = all_class_labels.reshape(-1)

        return all_class_data, all_class_labels
        
    def sample(
            self,
            device: str = "gpu"
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        
        self.key, subkey = jax.random.split(self.key)
        
        all_data, all_labels = self._sample_task_jit(
            subkey,
            self.class_indicies,
            self.class_lenghts,
            self.all_data,
            self.n_ways,
            self.samples_per_class
        )
        if device == "gpu":
            device = jax.devices('gpu')[0]
            all_data = jax.device_put(all_data, device)
            all_labels = jax.device_put(all_labels, device)
        
        data_reshaped = all_data.reshpe(self.n_ways, self.samples_per_class, *all_data.shape[1:])
        labels_reshaped = all_labels.reshape(self.n_ways, self.samples_per_class)

        support_data = data_reshaped[:,:self.k_shot].reshape(-1, *all_data.shape[1:])
        query_data = data_reshaped[:,self.k_shot:].reshape(-1, *all_data.shape[1:])

        support_labels = labels_reshaped[:, :self.k_shot].reshape(-1)
        query_labels = labels_reshaped[:, self.k_shot:].reshape(-1)

        return (support_data, support_labels) , (query_data, query_labels)
    
    def sample_batch(
            self,
            device:str = 'gpu',

    )-> tuple[tuple[Array, Array], tuple[Array, Array]]:
        self.key, *subkeys = jax.random.split(self.key, self.batch_size + 1)

        keys_array = jnp.stack(subkeys)

        all_data_batch, all_labels_batch = self._sample_batch_jit(
            keys_array,
            self.class_indicies,
            self.class_lenghts,
            self.all_data,
            self.n_ways,
            self.samples_per_class
        )

        if device == "gpu":
            device = jax.devices('gpu')[0]
            all_data_batch = jax.device_put(all_data_batch, device)
            all_labels_batch = jax.device_put(all_labels_batch, device)
        
        data_reshaped = all_data_batch.reshape(
            self.batch_size, self.n_ways, self.samples_per_class, *all_data_batch.shape[2:]
        )

        labels_reshaped = all_labels_batch.reshape(
            self.batch_size, self.n_ways, self.samples_per_class
        )

        support_data = data_reshaped[:, :, :self.k_shot].reshape(
            self.batch_size, -1, *all_data_batch.shape[2:]
        )
        query_data = data_reshaped[:, :, self.k_shot:].reshape(
            self.batch_size, -1, *all_data_batch.shape[2:]
        )

        support_labels = labels_reshaped[:, :, :self.k_shot].reshape(self.batch_size, -1)
        query_labels = labels_reshaped[:, :, self.k_shot:].reshape(self.batch_size, -1)

        return (support_data, support_labels), (query_data, query_labels)
    
    def get_memory_usage(self) -> dict:
        return {
            'all_data_device': self.all_data.device(),
            'class_indices_device': self.class_indicies.device(),
            'all_data_size_mb': self.all_data.nbytes / 1024 / 1024
        }