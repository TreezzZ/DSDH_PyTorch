import models.alexnet as alexnet
import models.vgg16 as vgg16

def load_model(arch, code_length):
    """
    Load cnn model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet.load_model(code_length)
    elif arch == 'vgg16':
        model = vgg16.load_model(code_length)
    else:
        raise ValueError('Invalid model name!')

    return model

