import random
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

class eval_mode:
    def __init__(self, *models):
        """
        Taking any number of models as arguments and storing them.
        """
        self.models = models

    def __enter__(self):
        """
        Creating an empty list to store 
        whether each model in th self.models list 
        is in training mode or in evaluation mode
        and setting the model to evaluation mode.
        
        By storing the training state of the models 
        before evaluation starts, we can restore the models
        back to their original state after the evaluation is done.
        """
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)
            
    def __exit__(self, *args):
        """
        Storing each model in the self.models list back to 
        it's original training state after the evaluation is done.
        """
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

    def set_seed_everywhere(seed):
        """
        Setting the same seed for PyTorch, NumPy, CUDA (if it is available), 
        and Python's random modules to ensure that all randomized operations 
        produce the exact same (sequence of) numbers every time the code runs. 
        This creates deterministic behavior and ensures 
        reproducibility and consistent results during evaluation of the model.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

def soft_update_params(net, target_net, tau):
    """
    Updating the parameters of the "target_net" to match
    the parameters of the "net". As a result of this soft update,
    the "target_net" will eventually converge to match the "net" in time.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def to_torch(xs, device):
    """
    Iterating through the each element in xs,
    converting it into a tensor, 
    moving the tensor to the specified device,
    collecting all these conversions in a tuple,
    and returning that tuple.
    """
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def weight_init(m):
    """
    Taking a neural network "m", 
    initializing the weights of the linear layers 
    using orthogonal initializaiton and
    initializing the biases as 0.
    
    If the layer is convolutional or transposed convolutional layer,
    it calculates a gain factor based on a ReLU activation, 
    applies orthogonal initialization with this gain factor 
    to the weights of the convolutional or transposed convolutional layer, 
    and initializes the biases to 0.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Until:
    def __init__(self, until, action_repeat=1):
        """
        Initializing the parameters that are used to 
        determine how long the process should continue (until)
        and how many times an action should be repeated (action_repeat).
        """
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        """
        If the value of the 'until' parameter is None, 
        we return True which means we should continue the process. 
        If we want to repeat the action 
        '_action_repeat' number of times, this means 
        '_until // _action_repeat' decision points. 
        If the current step is lower than this number, 
        we return True which means we should continue the process.
        Otherwise, we return False which means we should stop the process.
        """
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until

class Every:
    def __init__(self, every, action_repeat=1):
        """
        Initializing the parameters that are used to control
        how frequently an event should trigger (every)
        and how many times an action should be repeated (action_repeat).
        """
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        """
        Determining whether an event should be triggered at the current step.
        If the 'every' parameter is None, 
        the function returns False which means the event never triggers.
        Otherwise, it calculates how many agent decision points 
        correspond to the desired environment step frequency 
        by dividing 'every' by 'action_repeat'.
        If the current step is divisible by this adjusted frequency (step % every == 0), 
        the function returns True which means the event should trigger.
        Otherwise, it returns False which means the event should not trigger at this step.
        """
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:    
    def __init__(self):
        """
        Initializing variables that will be used to 
        track training time while excluding evaluation time.
        
        _start_time: Records the beginning of the experiment.
        _last_time: Records the last time reset() function is called.
        _eval_start_time: Records the beginning of the evaluation.
        _eval_time: Records the total time spent in evaluation mode.
        _eval_flag: Boolean value that indicates whether we are in evaluation mode.
        """
        self._start_time = time.time()
        self._last_time = time.time()
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def reset(self):
        """ 
        Measuring the time passed since 
        the last time reset() is called (elapsed_time), and 
        the time spent for training the model (total_time) and 
        returning both.
        """
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        """
        Moving to evaluation mode and
        setting the current time as the evaluation start time
        if we are not in evaluation mode.
        
        Adding the time spent since the evaluation starts 
        to the evaluation time, deactivating the evaluation mode, 
        and resetting the evaluation start time as 0 
        if we are already in evaluation mode and 
        this function is called second time.
        """
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        """
        Measuring the time spent since the beginning of the experiment 
        -- excluding the time spent for evaluation. In other words,
        this measures the time spent for training.
        """
        return time.time() - self._start_time - self._eval_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        """
        Initializing the lower bound ("low"), upper bound ("high"), and 
        an epsilon value that will be used to clamp 
        
        - any value lower than the lower bound 
        to the value of the (lower bound + epsilon),
        
        - any value higher than the upper bound 
        to the value of the (upper bound - epsilon)
        
        """
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        """ 
        Clamping the value of "x" to (lower bound + epsilon) 
        if it is lower than the lower bound or 
        clamping it to (upper bound - epsilon) 
        if it is larger than the upper bound.
        """
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        """
        Sampling a (sample_shape x batch_shape x event_shape) dimensional values 
        from Truncated Normal Distribution with 
        the mean value of 'loc', standard deviation value of 'scale',
        lower bound of 'low', upper bound of 'high', and epsilon value of 'eps'.
        
        - sample_shape: The number of times we want to draw from 
        the Truncated Normal Distribution. 
        - batch_shape: The number of Truncated Normal Distributions.
        - event_shape: the shape of each outcome from the distribution
        """
        
        # Concatenate sample_shape with the batch_shape and event_shape. 
        shape = self._extended_shape(sample_shape)
        
        # Generate a sample from a normal distribution with mean = 0 and std = 1.
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        
        # Scale the sample generated from the normal distribution 
        # in such a way that it has the desired standard deviation (self.scale)
        eps *= self.scale
        
        # Restrict how far the generated and scaled sample 'eps'
        # can deviate from 0.
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        
        # Take the center of the normal distribution (self.loc), 
        # add some noise that is generated from standard normal distribution,
        # scaled, and restricted to take a value between -clip and clip (eps), 
        # clamp this noisy center, and return it.
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    """
    Returning the float value of the 'schdl' if it can be converted into the float. 
    
    If the 'schdl' is in the form of 'linear(init, final, duration)': 
        - Divide 'duration' by the 'step' and ensure that the value (mix) 
        stays between 0 and 1. This tells us the total duration 
        that has elapsed with a value between 0 and 1.
            - When this value (mix) is 0, return  'init'. 
            - When this value (mix) is 1, return 'final'
            - When this value (mix) is between 0 and 1, return 
            the weighted average of these two values.
            
    If the 'schdl' is in the form of 'step_linear(init, fina1, duration1, final2, duration2)'
        - In the first phase (between 'init' and 'duration1'), 
        change the value from 'init' to 'final1' linearly. 
        - In the second phase (between 'duration1' and 'duration2'),
        change the value from 'final1' to 'final2' linearly. 
        
    If the 'schdl' does not match with any of these, raise and error 
    to indicate that the scheduling isn't implemented. 
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

class RandomShiftsAug(nn.Module):    
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        """
        Padding the image x, creating the coordinates of the image, 
        normalizing these coordinates, generating offsets, 
        normalizing these offsets, shifing the normalized coordinates with these offsets, 
        and retrieving the pixels from the original padded image 
        by using these normalized and shifted coordinates.
        """
        n, c, h, w = x.size()
        assert h == w
        
        # Pad the input image x to prevent losing content 
        # when shifting the image.
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        
        # Create a tensor that has values from "-1 + eps" to "1 - eps"
        # and that has "h + 2 * pad" values in total.
        # This tensor represents the normalized x coordinate values 
        # of the pixels in the input (image). 
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 
                                1.0 - eps,
                                h + 2 * self.pad,
                                device = x.device,
                                dtype = x.dtype)[:h]
        
        # Create a 2D normalized x and y coordinate values of the pixels in the input image
        # In the normalized coordinate system:
        # -1 represents leftmost pixel for x coordinate and topmost pixel for y coordinate. 
        # 1 represents the rightmost pixel for x coordinate and bottommost pixel for y coordinate. 
        # For each observation in the episode, base_grid contains 200 x 200 x 2 dimensional data.
        # This means that for each pixel, we have 2 variables that represent 
        # the normalized x and y coordinates of that pixel and 
        # we can use these coordinates to sample from in the input image.

        # For instance, at position base_grid[0, 100, 150], let's say that we have [-0.2, 0.4] values. 
        # This means that "for the output pixel at row 100, column 150 
        # sample from the source image at normalized position x=-0.2, y=0.4
        # which corresponds to a position slightly left and above the center of the image"        
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        # For each observation in the episode, 
        # generate 2 random numbers that will be used as offsets 
        # to shift the observation/image horizontally and vertically.        
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        
        # The shift/offsets above are in pixel space.
        # and they are in the range of 0 and 2 * pad.
        # Convert these pixel-space shifts into 
        # normalized coordinate space shifts.        
        shift *= 2.0 / (h + 2 * self.pad)

        # Shift the 2D normalized x and y coordinate values of the pixels in the input image
        # The grid below tells us where to sample the pixels from the input image        
        grid = base_grid + shift
        
        # Retrieve the pixel values from the padded input x 
        # by using the shifted and normalized x and y coordinate values of 
        # the pixels in x.
        x_shifted = F.grid_sample(x,
                                  grid,
                                  padding_mode='zeros',
                                  align_corners=False)
        
        # Return the shifted input image x
        return x_shifted