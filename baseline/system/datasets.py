import pandas as pd

import jaffadata as jd
from jaffadata.datasets import Arca23K, Arca23K_FSD


def dataset(name, arca23k_dir, fsd50k_dir, seed=None):
    if name.lower() == 'arca23k':
        return Arca23K(arca23k_dir, fsd50k_dir)
    if name.lower() == 'arca23k-fsd':
        return Arca23K_FSD(arca23k_dir, fsd50k_dir)
    if name.lower()[:8] == 'arca23k-':
        frac = float(name[8:])
        noisy = Arca23K(arca23k_dir, fsd50k_dir)
        clean = Arca23K_FSD(arca23k_dir, fsd50k_dir)
        clean_train_set = clean['training']
        noisy_train_set = noisy['training']
        train_set = mix_subsets(noisy_train_set, clean_train_set, frac, seed)
        noisy['training'] = train_set
        return noisy

    raise ValueError(f"Unrecognized dataset '{name}'")


def mix_subsets(noisy_train_set, clean_train_set, frac, seed=None):
    # Select proportion of examples from clean training set
    clean_train_set = sample(clean_train_set, 1 - frac, seed)
    target_sizes = clean_train_set.tags.groupby('label').size()

    def _sample(x):
        return x.sample(target_sizes[x.label[0]], random_state=seed)

    # Select proportion of examples from noisy training set
    df = noisy_train_set.tags.groupby('label').apply(_sample).droplevel(0)
    mask = ~noisy_train_set.tags.index.isin(df.index)
    noisy_train_set = noisy_train_set[mask]

    return jd.concat([noisy_train_set, clean_train_set], name='training')


def sample(subset, frac, seed=None):
    # Ensure at least 10 examples are sampled for each class
    df1 = subset.tags.groupby('label').sample(10, random_state=seed)
    n_examples = int(len(subset) * frac) - len(df1)
    mask = ~subset.tags.index.isin(df1.index)
    df2 = subset.tags[mask].sample(n_examples, random_state=seed)
    df = pd.concat([df1, df2])

    return subset.loc[df.index]
