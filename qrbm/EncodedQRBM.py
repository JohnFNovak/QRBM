import numpy as np
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

import tabu
from qrbm.persistent_sampler import Sampler
import sys

# from qrbm.MSQRBM import sigmoid

# TO FIGURE OUT:
# 1. should biases run from -1 to 1, or 0 to 1?
# 2. why aren't we clamping the visible biases during training?

class QRBM:
    def __init__(self,
                 data_template,
                 n_hidden: int,
                 err_function = 'mse',
                 qpu = False,
                 chain_strength: int = 5,
                 use_tqdm = True,
                 tqdm = None): # What's the type on 'tqdm'?

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.data_template = data_template
        self.n_visible = len(data_template.ravel())
        self.n_hidden = n_hidden
        self.qpu = qpu

        if qpu:
            self.sampler = Sampler(
                self.n_visible,
                self.n_hidden,
                sampler=EmbeddingComposite(DWaveSampler())
            )
        else:
            self.sampler = Sampler(
                self.n_visible,
                self.n_hidden,
                sampler=tabu.TabuSampler()
            )

        self.cs = chain_strength

        # Initialize weights randomly
        # Couplings
        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2) - 1
        # Visible biases
        self.visible_bias = (np.random.rand(self.n_visible))# * 2) - 1
        # Hidden biases
        self.hidden_bias = (np.random.rand(self.n_hidden))# * 2) - 1

        self.n_epoch = 0

    def get_weights(self):
        return self.w, \
               self.visible_bias, \
               self.hidden_bias

    def set_weights(self, w, visible_bias, hidden_bias):
        self.w = w
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias

    def train(self,
              training_data,
              epochs: int,
              lr: float = 0.1,
              lr_decay: float = 0.1,
              epoch_drop: Optional[int] = None,
              momentum = 0, # type?
              batch_size: Optional[int] = None):

        learning_curve_plot = []

        if epoch_drop == None:
            epoch_drop = int(epochs / 5)

        # initial momentum velocity value
        momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
        momentum_v = np.zeros(len(self.visible_bias))
        momentum_h = np.zeros(len(self.hidden_bias))

        # for epoch in range(epochs):
        for epoch in self.tqdm(range(epochs)):
            # Single step
            # 1
            # 1.1 Take a training sample v
            random_selected_training_data_idx = np.random.randint(0, len(training_data))
            if epoch % 1000 == 0:
                print(epoch)
                print(random_selected_training_data_idx)

            # The visible layer values is the input data, plus the encoded labels
            v = training_data[random_selected_training_data_idx].ravel()
            mask = (v > 0).astype(int)
            # print(v)
            # Handle the mask:
            # We zero out entries that are less than zero and set them to the default visible bias
            v = v * (v > 0).astype(int) + self.visible_bias * (v < 0).astype(int)
            # print(v)
            old_v = v

            # I don't understand the batch_size logic
            if batch_size is not None:
                if epoch % batch_size != 0:
                    # NOTE: v_prim is defined further on
                    old_v = v_prim

            # 1.2 Compute the probabilites of the hidden units
            # persisntent CD takes v from previous iterations
            h = self.sampler.sample_hidden(old_v,
                                           self.hidden_bias,
                                           self.w,
                                           chain_strength=self.cs)

            # print(h)
            energy = - (v@self.visible_bias) - (h@self.hidden_bias) - (v@(self.w@h))
            # print(energy, (v@self.visible_bias), (h@self.hidden_bias), (v@(self.w@h)))
            # 2 Compute the outer product of v and h and call this the positive gradient
            pos_grad = np.outer(v, h)
            # print(pos_grad)

            # 3
            # 3.1 From h, sample a reconstruction v' of the visible units
            v_prim = self.sampler.sample_visible(self.visible_bias,
                                                 h,
                                                 self.w.T,
                                                 chain_strength=self.cs)
            # print(v_prim)
            # NOTE: should we apply the mask to this one too? I don't think so...

            # 3.2 Then resample the hidden activations h' from this. (Gibbs sampling step)
            h_prim = self.sampler.sample_hidden(v_prim,
                                                self.hidden_bias,
                                                self.w,
                                                chain_strength=self.cs)
            # print(h_prim)

            # 4 Compute the outer product of v' and h' and call this the negative gradient
            neg_grad = np.outer(v_prim, h_prim)
            # print(neg_grad)

            # 5 Let the update to the weight matrix W be the positive graidnet minus the negaitve gradient
            #   times some learning rate
            update_w = pos_grad - neg_grad
            # print(update_w)
            momentum_w = (momentum * momentum_w) + (lr * update_w)

            # I probably need to do something with the mask here...
            self.w += momentum_w
            # # Restrict to [-1, 1)
            # self.w = (self.w * (self.w > -1).astype(int)) - (self.w < -1).astype(int)
            # self.w = (self.w * (self.w < 1).astype(int)) + (self.w > 1).astype(int)

            # 6 Update the biases a and b analogously: a=epsilon*(v-v'), b=epsilon*(h-h')
            momentum_v = momentum * momentum_v + lr * (np.array(v) - np.array(v_prim))
            momentum_h = momentum * momentum_h + lr * (np.array(h) - np.array(h_prim))

            self.visible_bias += momentum_v
            self.hidden_bias += momentum_h
            # Restrict to [0, 1)
            self.visible_bias = self.visible_bias * (self.visible_bias > 0).astype(int)
            self.visible_bias = (self.visible_bias * (self.visible_bias < 1).astype(int)) + (self.visible_bias > 1).astype(int)
            self.hidden_bias = self.hidden_bias * (self.hidden_bias > 0).astype(int)
            self.hidden_bias = (self.hidden_bias * (self.hidden_bias < 1).astype(int)) + (self.hidden_bias > 1).astype(int)

            if epoch % epoch_drop == (epoch_drop-1):
                # learning rate decay
                lr *= (1 - lr_decay)

            sample_v = v
            sample_h = self.sampler.sample_hidden(sample_v,
                                                  self.hidden_bias,
                                                  self.w,
                                                  chain_strength=self.cs)
            sample_output = self.sampler.sample_visible(self.visible_bias,
                                                        sample_h,
                                                        self.w.T,
                                                        chain_strength=self.cs)
            # error = np.sum((np.abs(np.array(v) - np.array(sample_output))*mask))
            error = np.sum((np.abs(np.array(v) - np.array(sample_output))**2))
            # print(error)
            learning_curve_plot.append(error)
            del(sample_output)

        plt.figure()
        plt.plot(learning_curve_plot)
        plt.xlabel('epoch')
        plt.ylabel('normalised MSE')
        plt.show()
        return

    def sample_hidden(self, visible, n_times=1, mask=None, use_visible=False):
        sample_v = visible.copy()
        if mask is not None and use_visible:
            sample_v[mask] = self.visible_bias[mask]
            sample_h = self.sampler.sample_hidden(sample_v,
                                                self.hidden_bias,
                                                self.w,
                                                chain_strength=self.cs)
        else:
            sample_h = self.sampler.sample_hidden(sample_v,
                                                self.hidden_bias,
                                                self.w,
                                                chain_strength=self.cs,
                                                mask=mask)
        if n_times > 1:
            for _ in range(n_times - 1):
                sample_v = self.sampler.sample_visible(self.visible_bias,
                                                       sample_h,
                                                       self.w.T,
                                                       chain_strength=self.cs)
                sample_h = self.sampler.sample_hidden(sample_v,
                                                      self.hidden_bias,
                                                      self.w,
                                                      chain_strength=self.cs)
        return sample_h


    def save(self, filename):
        with np.printoptions(threshold=sys.maxsize):
            parameters = [str(self.n_hidden),
                          str(self.n_visible),
                          np.array_repr(self.visible_bias),
                          np.array_repr(self.hidden_bias),
                          np.array_repr(self.w)]
            with open(filename, 'w') as file:
                file.write('#'.join(parameters))


    def load(self, filename):
        with open(filename) as file:
            res = file.read()
            parameters = res.split('#')
            self.n_hidden = eval(parameters[0])
            self.n_visible = eval(parameters[1])
            self.visible_bias = eval('np.'+parameters[2])
            self.hidden_bias = eval('np.'+parameters[3])
            self.w = eval('np.'+parameters[4])
