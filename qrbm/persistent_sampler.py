from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from pyqubo import Binary, Placeholder
import tabu

class Sampler:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 qpu=False,
                 sampler=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # choose if you want to use real QPU or local simulation
        if sampler is None:
            if qpu:
                self.sampler = EmbeddingComposite(DWaveSampler())
            else:
                self.sampler = tabu.TabuSampler()
        else:
            self.sampler = sampler

        # visible -> hidden Hamiltonian
        H = 0
        H_vars = []

        # initialize all variables (one for each node in the opposite layer)
        for j in range(n_hidden):
            H_vars.append(Binary(str(j)))
            b = Placeholder(f'bh{j}')
            H += -1 * b * H_vars[j]

        for i in range(n_visible):
            b = Placeholder(f'bv{i}')
            # add reward to every connection
            for j in range(n_hidden):
                w = Placeholder(f'W{i},{j}')
                H += -1 * w * H_vars[j] * b

        self.vhH_model = H.compile()

        # hidden -> visible Hamiltonian
        H = 0
        H_vars = []

        # initialize all variables (one for each node in the opposite layer)
        for j in range(n_visible):
            H_vars.append(Binary(str(j)))
            b = Placeholder(f'bv{j}')
            H += -1 * b * H_vars[j]

        for i in range(n_hidden):
            b = Placeholder(f'bh{i}')
            # add reward to every connection
            for j in range(n_visible):
                w = Placeholder(f'W{i},{j}')
                H += -1 * w * H_vars[j] * b

        self.hvH_model = H.compile()

    def sample_hidden(self,
                      v_biases,
                      h_biases,
                      weights,
                      chain_strength=2,
                      num_reads=1,
                      mask=None):
        # prepare feed_dict
        feed_dict = {}
        for i in range(self.n_visible):
            feed_dict[f'bv{i}'] = v_biases[i]
            if mask is not None and mask[i]:
                for j in range(self.n_hidden):
                    feed_dict[f'W{i},{j}'] = 0
            else:
                # add reward to every connection
                for j in range(self.n_hidden):
                    feed_dict[f'W{i},{j}'] = weights[i][j]

        for j in range(self.n_hidden):
            feed_dict[f'bh{j}'] = h_biases[j]

        bqm = self.vhH_model.to_bqm(feed_dict=feed_dict)
        sampleset = self.sampler.sample(bqm, chain_strength=chain_strength, num_reads=num_reads)
        solution1 = sampleset.first.sample
        solution1_list = [(k, v) for k, v in solution1.items()]
        solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
        solution1_list_final = [v for (k, v) in solution1_list]
        return solution1_list_final

    def sample_visible(self,
                       v_biases,
                       h_biases,
                       weights,
                       chain_strength=2,
                       num_reads=1,
                       mask=None):
        # prepare feed_dict
        feed_dict = {}
        for i in range(self.n_hidden):
            feed_dict[f'bh{i}'] = h_biases[i]
            if mask is not None and mask[i]:
                for j in range(self.n_visible):
                    feed_dict[f'W{i},{j}'] = 0
            else:
                # add reward to every connection
                for j in range(self.n_visible):
                    feed_dict[f'W{i},{j}'] = weights[i][j]

        for j in range(self.n_visible):
            feed_dict[f'bv{j}'] = v_biases[j]

        bqm = self.hvH_model.to_bqm(feed_dict=feed_dict)
        sampleset = self.sampler.sample(bqm, chain_strength=chain_strength, num_reads=num_reads)
        solution1 = sampleset.first.sample
        solution1_list = [(k, v) for k, v in solution1.items()]
        solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
        solution1_list_final = [v for (k, v) in solution1_list]
        return solution1_list_final
