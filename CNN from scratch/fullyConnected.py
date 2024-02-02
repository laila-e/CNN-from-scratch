import numpy as np


class ANN:

    def __init__(self, input_len, nodes):
        self.param = {}
        self.dparam = {}
        # self.param["weight"] = np.random.randn(input_len, nodes) / input_len
        # self.param["biases"] = np.zeros(nodes)
        self.param["weight"]=np.load(r"D:\utilisateurs\DELL\Desktop\leila\Master\S1\Mr jamal\lolo.py\weight.npy")
        self.param["biases"]=np.load(r"D:\utilisateurs\DELL\Desktop\leila\Master\S1\Mr jamal\lolo.py\biases.npy")
    def forward(self, input):

        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.param["weight"].shape

        totals = np.dot(input, self.param["weight"]) + self.param["biases"]
        self.last_totals = totals

        
        max = np.max(totals)
        exp = np.exp(totals-max)
        return exp/ np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):

        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.param["weight"]

            d_L_d_t = gradient * d_out_d_t

            self.dparam["weight"] = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            self.dparam["biases"] = d_L_d_t * d_t_d_b

            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # self.param["weight"] -= learn_rate * d_L_d_w
            # self.param["biases"] -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)
