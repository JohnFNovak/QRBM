import copy
import operator
import random

import sys
import matplotlib.pyplot as plt
import numpy as np

# from qrbm.oldsampler import Sampler
# import qrbm.sampler as samp
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

import tabu
from qrbm.persistent_sampler import Sampler

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

class MSQRBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 err_function='mse',
                 qpu=False,
                 chain_strength=5,
                 use_tqdm=True,
                 tqdm=None,
                 result_picture_tab = None):

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
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

        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 1
        # Visible biases
        self.visible_bias = (np.random.rand(self.n_visible))# * 2) - 1
        # Hidden biases
        self.hidden_bias = (np.random.rand(self.n_hidden))# * 2) - 1

        #docelowo własny sampler ogarnąć
        # self.sampler = Sampler()

        self.n_epoch = 0
        self.result_picture_tab = result_picture_tab

    def get_weights(self):
        return self.w, \
               self.visible_bias, \
               self.hidden_bias

    def set_weights(self, w, visible_bias, hidden_bias):
        self.w = w
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias

    # #budowane jest qubo inaczej tego chyba nie da rady zrobić
    # def set_qubo(self):
    #     visible_bias = self.visible_bias
    #     hidden_bias = self.hidden_bias
    #     w = self.w
    #
    #     Q = {}
    #     for i in range(self.n_visible):
    #         Q[(i, i)] = -1 * visible_bias[i]
    #     for j in range(self.n_hidden):
    #         Q[(j + self.n_visible, j + self.n_visible)] = -1 * hidden_bias[j]
    #     for i in range(self.n_visible):
    #         for j in range(self.n_hidden):
    #             Q[(i, self.n_visible + j)] = -1 * w[i][j]
    #     self.Q = Q
    #     # print("qubo: ", self.Q)
    #
    # def get_Z(self):
    #     Z = np.sum(np.exp(-1 * self.energies))
    #     self.Z = Z
    #     return Z
    #
    #
    # # tutaj bym ogarnął to sam
    # def sample_qubo(self, num_samps=100):
    #     if not hasattr(self, 'Q'):
    #         self.set_qubo()
    #     self.samples, self.energies, self.num_occurrences = self.sampler.sample_qubo(self.Q, num_samps=num_samps)
    #     # self.samples, self.energies, self.num_occurrences = self.sampler.sample_qubo(self.Q, num_samps=1)
    #     self.energies /= np.max(np.abs(self.energies))
    #     self.get_Z()
    #     return self.samples

    def train(self, training_data, len_x=1, len_y=1, epochs=50, lr=0.1, lr_decay=0.1, epoch_drop = None, momentum = 0, batch_size = None):
        """
            maximize the product of probabilities assigned to some training set V
            optimize the weight vector

            single-step contrastive divergence (CD-1):
            1. Take a training sample v,
                compute the probabilities of the hidden units and
                sample a hidden activation vector h from this probability distribution.
            2. Compute the outer product of v and h and call this the positive gradient.
            3. From h, sample a reconstruction v' of the visible units,
                then resample the hidden activations h' from this. (Gibbs sampling step)
            4. Compute the outer product of v' and h' and call this the negative gradient.
            5. Let the update to the weight matrix W be the positive gradient minus the negative gradient,
                times some learning rate
            6. Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')

            https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine
        """
        learning_curve_plot = []

        if epoch_drop == None:
            epoch_drop = epochs / 5

        # initial momentum velocity value
        momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
        momentum_v = np.zeros(len(self.visible_bias))
        momentum_h = np.zeros(len(self.hidden_bias))

        for epoch in self.tqdm(range(epochs)):
            # single step
            # print("Training data len", len(training_data))
            # 1
            # 1.1 Take a training sample v
            random_selected_training_data_idx = epoch % len(training_data)
            # print("selected_training_data_idx: ", random_selected_training_data_idx)

            v = training_data[random_selected_training_data_idx]
            old_v = v

            if batch_size is not None:
                if epoch % batch_size != 0:
                    old_v = v_prim

            # # 1.2 compute the probabilities of the hidden units
            # prob_h = sigmoid(self.hidden_bias + np.dot(v, self.w))
            # print(self.hidden_bias)
            # print(v)
            # print(self.w)

            # h należy teraz wysamplować z tych wartości prawdopodobieństw
            #klasycznie mogę to zrobić tak:
            # print("self.hidden_bias: ", self.hidden_bias)
            # print("np.dot(v, self.w): ", np.dot(v, self.w))
            # print("self.hidden_bias + np.sum(np.dot(v, self.w)): ", self.hidden_bias + np.dot(v, self.w))
            # print("prob_h: ", prob_h)
            # h = (np.random.rand(len(self.hidden_bias)) < prob_h).astype(int)

            # persisntent CD takes v from previous iterations
            # h = samp.sample_opposite_layer_pyqubo(old_v, self.visible_bias,
            #                                       self.w, self.hidden_bias,
            #                                       qpu=self.qpu,
            #                                       chain_strength=self.cs)
            h = self.sampler.sample_hidden(old_v,
                                           self.hidden_bias,
                                           self.w,
                                           chain_strength=self.cs)
            # h = samp.sample_opposite_layer_pyqubo(v, self.visible_bias, self.w, self.hidden_bias)

            # print("h: ", h)

            # 2 Compute the outer product of v and h and call this the positive gradient.
            pos_grad = np.outer(v, h)
            # print("pos_grad:", pos_grad)

            # 3
            # 3.1 From h, sample a reconstruction v' of the visible units,
            # prob_v_prim = sigmoid(self.visible_bias + np.dot(h, self.w.T))
            # #znów klasycznie
            # v_prim = (np.random.rand(len(self.visible_bias)) < prob_v_prim).astype(int)

            # v_prim = samp.sample_opposite_layer_pyqubo(h, self.hidden_bias,
            #                                            self.w.T,
            #                                            self.visible_bias,
            #                                            qpu=self.qpu,
            #                                            chain_strength=self.cs)
            v_prim = self.sampler.sample_visible(self.visible_bias,
                                                 h,
                                                 self.w.T,
                                                 chain_strength=self.cs)

            # print("v_prim: ", v_prim)

            # 3.2 then resample the hidden activations h' from this. (Gibbs sampling step)
            # prob_h_prim = sigmoid(self.hidden_bias + np.dot(v_prim, self.w))
            # # h należy teraz wysamplować z tych wartości prawdopodobieństw
            # # klasycznie mogę to zrobić tak:
            # h_prim = (np.random.rand(len(self.hidden_bias)) < prob_h_prim).astype(int)

            # h_prim = samp.sample_opposite_layer_pyqubo(v_prim,
            #                                            self.visible_bias,
            #                                            self.w, self.hidden_bias,
            #                                            qpu=self.qpu,
            #                                            chain_strength=self.cs)
            h_prim = self.sampler.sample_hidden(v_prim,
                                                self.hidden_bias,
                                                self.w,
                                                chain_strength=self.cs)
            # print("h_prim: ", h_prim)

            # 4 Compute the outer product of v' and h' and call this the negative gradient.
            neg_grad = np.outer(v_prim, h_prim)
            # print("neg_grad:", neg_grad)

            # 5 Let the update to the weight matrix W be the positive gradient minus the negative gradient,
            #        times some learning rate
            #this is for momentum (default value 0 doesn't change anything)

            momentum_w = momentum * momentum_w + lr * (pos_grad - neg_grad)

            self.w += momentum_w
            # print("w: ", self.w)

            # 6 Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')
            #momentum here

            momentum_v = momentum * momentum_v + lr * (np.array(v) - np.array(v_prim))
            momentum_h = momentum * momentum_h + lr * (np.array(h) - np.array(h_prim))

            self.visible_bias += momentum_v
            self.hidden_bias += momentum_h
            # print("visible_bias: ", self.visible_bias)
            # print("hidden_bias: ", self.hidden_bias)
            # Restrict to [0, 1)
            self.visible_bias = self.visible_bias * (self.visible_bias > 0).astype(int)
            self.visible_bias = (self.visible_bias * (self.visible_bias < 1).astype(int)) + (self.visible_bias > 1).astype(int)
            self.hidden_bias = self.hidden_bias * (self.hidden_bias > 0).astype(int)
            self.hidden_bias = (self.hidden_bias * (self.hidden_bias < 1).astype(int)) + (self.hidden_bias > 1).astype(int)


            # po updacie musimy zaktualizować qubo
            # self.set_qubo()

            if epoch % epoch_drop == (epoch_drop-1):
                # krzywa uczenia
                # sample_v = v
                # prob_sample_h = sigmoid(self.hidden_bias + np.dot(v, self.w))
                # sample_h = (np.random.rand(len(self.hidden_bias)) < prob_sample_h).astype(int)
                # prob_sample_v_out = sigmoid(self.visible_bias + np.dot(sample_h, self.w.T))
                # sample_output = (np.random.rand(len(self.visible_bias)) < prob_sample_v_out).astype(int)
                # learning_curve_plot.append(np.sum((np.array(v) - np.array(sample_output)) ** 2))

                #learning_rate_decay
                lr *= (1 - lr_decay)
                # print("lr = ", lr)

            #krzywa uczenia
            # sample_v = samp.sample_v(self.visible_bias)
            sample_v = v
            # sample_h = samp.sample_opposite_layer_pyqubo(sample_v,
            #                                              self.visible_bias,
            #                                              self.w,
            #                                              self.hidden_bias,
            #                                              qpu=self.qpu,
            #                                              chain_strength=self.cs)
            # sample_output = samp.sample_opposite_layer_pyqubo(sample_h,
            #                                                   self.hidden_bias,
            #                                                   self.w.T,
            #                                                   self.visible_bias,
            #                                                   qpu=self.qpu,
            #                                                   chain_strength=self.cs)
            sample_h = self.sampler.sample_hidden(sample_v,
                                                  self.hidden_bias,
                                                  self.w,
                                                  chain_strength=self.cs)
            sample_output = self.sampler.sample_visible(self.visible_bias,
                                                        sample_h,
                                                        self.w.T,
                                                        chain_strength=self.cs)
            learning_curve_plot.append(np.sum((np.array(v) - np.array(sample_output))**2))


        #koniec
        plt.figure()
        plt.plot(learning_curve_plot)
        plt.xlabel('epoch')
        plt.ylabel('normalised MSE')
        plt.show()
        return

    def generate(self, test_img = None):
        sample_v = []
        if test_img == None:
            # sample_v = samp.sample_v(self.visible_bias, qpu=self.qpu,
            #                          chain_strength=self.cs)
            sample_v = self.sampler.sample_visible(self.visible_bias,
                                                   sample_h,
                                                   self.w.T,
                                                   chain_strength=self.cs)
        else:
            sample_v = test_img
        # sample_h = samp.sample_opposite_layer_pyqubo(sample_v,
        #                                              self.visible_bias, self.w,
        #                                              self.hidden_bias,
        #                                              qpu=self.qpu,
        #                                              chain_strength=self.cs)
        sample_h = self.sampler.sample_hidden(sample_v,
                                                self.hidden_bias,
                                                self.w,
                                                chain_strength=self.cs)
        # sample_output = samp.sample_opposite_layer_pyqubo(sample_h,
        #                                                   self.hidden_bias,
        #                                                   self.w.T,
        #                                                   self.visible_bias,
        #                                                   qpu=self.qpu,
        #                                                   chain_strength=self.cs)
        sample_output = self.sampler.sample_visible(self.visible_bias,
                                                   sample_h,
                                                   self.w.T,
                                                   chain_strength=self.cs)
        return sample_output


    def evaluate(self, result, test_img = None):
        # sample_output = self.generate(test_img = test_img)
        min_sum = 1000000
        for pic in self.result_picture_tab:
            new_sum = np.sum((np.array(result) - np.array(pic)) ** 2)
            if new_sum < min_sum:
                min_sum = new_sum

        return min_sum

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
