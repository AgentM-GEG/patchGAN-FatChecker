import torch
from torch import nn
import functools
from torch.nn.parameter import Parameter


class Transferable():
    def __init__(self):
        super(Transferable, self).__init__()

    def load_transfer_data(self, checkpoint, verbose=False):
        state_dict = torch.load(checkpoint, map_location=next(self.parameters()).device)
        own_state = self.state_dict()
        state_names = list(own_state.keys())
        count = 0
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            # find the weight with the closest name to this
            sub_name = '.'.join(name.split('.')[-2:])
            own_state_name = [n for n in state_names if sub_name in n]
            if len(own_state_name) == 1:
                own_state_name = own_state_name[0]
            else:
                if verbose:
                    print(f'{name} not found')
                continue

            if param.shape == own_state[own_state_name].data.shape:
                own_state[own_state_name].copy_(param)
                count += 1

        if count == 0:
            print("WARNING: Could not transfer over any weights!")
        else:
            print(f"Loaded weights for {count} layers")


def get_norm_layer():
    """Return a normalization layer
       For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    """
    norm_type = 'batch'
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer


# custom weights initialization called on generator and discriminator
# scaling here means std
def weights_init(net, init_type='normal', scaling=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
