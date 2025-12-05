import jax
import jax.numpy as jnp
import random

class Sample_Task:
    def __init__(self, dataset, key, n_ways=5, k_shot=1, q_query = 15) -> None:
        self.class_to_indices = {}

        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.key=key
        self.n_ways=n_ways
        self.k_shot=k_shot
        self.q_query=q_query

        self.dataset = dataset

    def sample(self):
        selected_classes = jax.random.choice(key=self.key, a= jnp.array(list(self.class_to_indices.keys()), dtype=jnp.int32), shape=(1, self.n_ways), replace=False)[0]
        self.key, key1 = jax.random.split(self.key)
        

        support_data = []
        query_data = []

        support_target = []
        query_target = []

        for new_label, original_class in enumerate(selected_classes):
            indicies = self.class_to_indices[original_class.item()]

            indicies = jax.random.choice(key=key1, a=jnp.array(indicies), shape=(1, len(indicies)))[0]
            key1, _ = jax.random.split(key1)
            
            # random.shuffle(indicies)

            support_data.extend([self.dataset[idx][0].numpy() for idx in indicies[:self.k_shot]])
            query_data.extend([self.dataset[idx][0].numpy() for idx in indicies[self.k_shot:self.k_shot + self.q_query]])

            support_target.extend([new_label] * self.k_shot)
            query_target.extend([new_label] * self.q_query)
        
        support_set = (jnp.stack(support_data), jnp.stack(support_target))
        query_set = (jnp.stack(query_data), jnp.stack(query_target))
    
        return support_set, query_set
