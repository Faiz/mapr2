import tensorflow as tf
import numpy as np
from collections import defaultdict
from functools import partial
from policy import NN
from dist import DiagGaussianPd


class PGAgent:
    def __init__(self, id, policy_output_num=2, opponent_model_output_num=2, action_range=10., alpha=0.001, beta=0.001):
        self.id = id
        self.policy = NN(policy_output_num)
        self.opponent_model = NN(opponent_model_output_num)
        self.buffer = []
        self.alpha = alpha
        self.beta = beta

        self.flat_m = None
        self.flat_c = None
        self.flat_opp = None

        self.P_s = None
        self.action_range = action_range

    def _tfarray(self, arr):
        return tf.convert_to_tensor(arr)

    def get_policy_dist(self, state, opp_action):
        input = tf.reshape(tf.stack([state, opp_action], axis=1), [-1, 2])
        flat = self.policy(input)
        self.flat_c = DiagGaussianPd(flat)
        return self.flat_c

    def get_opponent_model_dist(self, state):
        flat = self.opponent_model(state)
        self.flat_opp = DiagGaussianPd(flat)
        return self.flat_opp

    def compute_marginal_policy(self, state):
        state = tf.reshape(state, [-1, 1])        
        flat_opp = self.get_opponent_model_dist(state)
        flat_p = self.get_policy_dist(state, flat_opp.sample())
        mu1, mu2, sig1, sig2 = flat_opp.mean, flat_p.mean, flat_opp.std, flat_p.std
        num1 = tf.multiply(mu1, tf.pow(sig2, 2))
        num2 = tf.multiply(mu2, tf.pow(sig1, 2))
        num = tf.add(num1, num2)
        denom = tf.add(tf.pow(sig2, 2), tf.pow(sig1, 2))
        mu = tf.divide(num, denom)
        num = tf.multiply(tf.pow(sig1, 2), tf.pow(sig2, 2))
        denom = tf.add(tf.pow(sig1, 2), tf.pow(sig2, 2))
        sigma = tf.divide(num, denom)
        flat_m = tf.reshape(tf.stack([mu, tf.log(sigma)], axis=1), [-1, 2])
        self.flat_m = DiagGaussianPd(flat_m)
        return self.flat_m

    def act(self, state):
        action = self.compute_marginal_policy(state).sample()
        return tf.scalar_mul(self.action_range, tf.tanh(action))

    def save_history(self, event):
        self.buffer[-1].append(event)

    def start_new_batch(self):
        self.buffer.append([])

    def update_P(self, window_length):
        window_length = min(len(self.buffer), window_length)
        m = []
        # since there is only one state
        for mini_batch in self.buffer[-window_length:]:
            for _, _, a_opp, _, _ in mini_batch:
                m.append(a_opp)
        flat = np.array([np.mean(m), tf.log(np.std(m))])
        self.P_s = DiagGaussianPd(flat)

        
    def loss_1(self, s, a, a_i, r):
        self.flat_m = self.compute_marginal_policy(s)
        phi_div_tsi = tf.divide(self.flat_opp.prob(a_i), self.P_s.prob(a_i))
        mult_1 = tf.multiply(phi_div_tsi, self.flat_c.logp(a))
        first = tf.multiply(mult_1, r)

        second_1 = self.flat_m.logp(a)
        second_2 = self.flat_m.logp(a)
        second_2 = tf.stop_gradient(second_2)
        second_2 = tf.add(second_2, 1)
        second = tf.multiply(second_1, second_2)
        return tf.reduce_mean(tf.add(first, second))

    def loss_2(self, s, a, a_i, r):
        self.flat_m = self.compute_marginal_policy(s)
        first_1 = tf.divide(self.flat_opp.prob(a_i), self.P_s.prob(a_i))
        first_1 = tf.multiply(first_1, self.flat_opp.logp(a_i))
        first_2 = tf.add(r, -tf.multiply(tf.exp(self.flat_c.logp(a)), self.flat_c.logp(a)))
        first = tf.multiply(first_1, first_2)
        second = self.flat_opp.kl(self.P_s)
        return tf.reduce_mean(tf.subtract(first, second))

    def update_params(self):
        batch = np.array(self.buffer[-1])

        states = batch[:, 0].reshape(-1, 1)
        actions = batch[:, 1].reshape(-1, 1)
        opp_actions = batch[:, 2].reshape(-1, 1)
        rewards = batch[:, 4].reshape(-1, 1)

        theta_variables = self.policy.get_trainable()

        loss_1_func = partial(
            self.loss_1,
            states, actions, opp_actions, rewards
        )

        tf.train.AdamOptimizer(
            learning_rate=self.alpha).minimize(loss_1_func, var_list=theta_variables)
        loss_2_func = partial(
            self.loss_2,
            states, actions, opp_actions, rewards
        )
        phi_variables = self.opponent_model.get_trainable()
        tf.train.AdamOptimizer(
            learning_rate=self.beta).minimize(loss_2_func, var_list=phi_variables)
