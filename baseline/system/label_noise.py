import numpy as np

import jaffadata as jd


class LabelNoise:
    def __init__(self, noise_type, noise_rate, **kwargs):
        if noise_type not in ['uniform', 'class']:
            raise ValueError(f"Unrecognized noise type: '{noise_type}'")
        if not isinstance(noise_rate, (int, float)) \
                and (noise_rate < 0 or noise_rate > 1):
            raise ValueError('Noise rate must be a number in [0,1]')

        self.noise_type = noise_type
        if self.noise_type == 'class':
            self.p = kwargs['p']
        self.noise_rate = noise_rate

    def relabeler(self, subset):
        # Map labels to integers in {1,...,K}
        subset = subset[:len(subset)]  # Create copy
        subset.tags['label_idx'] = subset.tags.label.cat.codes

        # Randomly select a fraction of the examples to corrupt
        corrupt_idx = subset.tags.sample(frac=self.noise_rate).index

        # Corrupt the labels of the selected examples
        n_labels = len(subset.dataset.label_set)
        if self.noise_type == 'uniform':
            offsets = np.random.randint(1, n_labels, len(corrupt_idx))
        elif self.noise_type == 'class':
            offsets = np.random.geometric(self.p, len(corrupt_idx))
        subset.tags.loc[corrupt_idx, 'label_idx'] = \
            (subset.tags['label_idx'].loc[corrupt_idx] + offsets) % n_labels

        def _target(_, index=None):
            return jd.binarize(subset, 'label_idx', index, is_label=False)

        return _target
