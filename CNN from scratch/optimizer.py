import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialisation(self, param, type_optimizer):
        if type_optimizer == 'adam':
            m = {}
            v = {}
            for k in param:
                m[k] = np.zeros(param[k].shape)
                v[k] = np.zeros(param[k].shape)
            return m, v

    def adam(self, x, dx, m, v):
        for k in x:
            m[k] = self.beta1*m[k]+(1-self.beta1)*dx[k]
            v[k] = self.beta2*v[k]+(1-self.beta2)*np.square(dx[k])
            m_hot = m[k]/(1-self.beta1)
            v_hot = v[k]/(1-self.beta2)
            x[k] = x[k]-self.learning_rate*m_hot/(np.sqrt(v_hot)+self.epsilon)
        return x
