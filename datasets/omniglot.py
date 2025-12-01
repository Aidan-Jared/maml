import numpy as np
import json
import concurrent.futures

from glob import glob
from PIL import Image
from collections import defaultdict

import functional as F
from base import MetaDataset
from util import download_url, get_asset

class Omniglot(MetaDataset):
    name = 'omniglot'
    url = 'https://raw.githubusercontent.com/brendenlake/omniglot/master/python'
    filenames = ['images_background', 'images_evaluation']
    shape = (28, 28, 1)

    def __init__(self, root, batch_size, shots=5, ways=5, test_shots=None, size=None, split='train', seed=42, download=False):
        super().__init__(root, batch_size, shots, ways, test_shots, size, split, seed, download)
        self.load_data()
    
    def load_data(self):
        if self._data is None:
            arrays, labels2indices = [], dict()
            offset = 0

            for data_filename, labels_filename in self.split_filenames:
                with open(data_filename, 'rb') as f:
                    data = np.load(f)
                    arrays.append(data)

                with open(labels_filename, 'r') as f:
                    labels = json.load(f)

                    labels2indices.update({label: offset + np.array(indicies) for label, indicies in labels.items()})

                    offset += data.shape[0]
            arrays = np.concat(arrays, axis=0)

            data, labels2indices_aug, num_samples = [], {}, arrays.shape[0]

            for k in [0, 1, 2, 3]