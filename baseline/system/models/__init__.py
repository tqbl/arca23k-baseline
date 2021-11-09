from .vgg import VGGa, VGG9a, VGG11a


__all__ = [
    'VGGa',
    'VGG9a',
    'VGG11a',
    'create_model',
]


def create_model(model_name, n_channels, n_classes, **kwargs):
    model_classes = {
        'vgg9a': VGG9a,
        'vgg11a': VGG11a,
    }

    try:
        model_class = model_classes[model_name]
        model = model_class(n_channels, n_classes, **kwargs)
    except KeyError:
        raise ValueError(f'Unrecognized model type: {model_name}')

    return model
