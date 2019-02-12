import tensorflow as tf
import numpy as np
from collections import defaultdict
from functools import partial

class NN:
    def __init__(self, input, output):
        init = tf.contrib.layers.xavier_initializer
        self.hidden_1 = tf.layers.Dense(units=32, activation=tf.nn.relu, bias_initializer=init())
        self.hidden_2 = tf.layers.Dense(units=32, activation=tf.nn.relu, bias_initializer=init())
        self.output = tf.layers.Dense(units=output, activation=tf.nn.relu, bias_initializer=init())
    
    def __call__(self, x):
        return self.output(
            self.hidden_2(
                self.hidden_1(x)
            )
        )

    def get_trainable(self):
        weights = []
        for layer in [self.hidden_1, self.hidden_2, self.output]:
            weights.extend(layer.trainable_variables)
        return weights

class PGAgent:
    def __init__(self, id, policy=NN(2, 2), opp_model=NN(1, 2), alpha=0.001, beta=0.001):
        self.id = id
        self.policy = policy
        self.opponent_model = opp_model
        self.buffer = []
        self.P_s = defaultdict(partial(list, [0, 0]))
        self.alpha = alpha
        self.beta = beta

    def _tfarray(self, arr):
        return tf.convert_to_tensor(arr)

    def get_policy_params(self, state, opp_action):
        return tf.reshape(
            self.policy(self._tfarray(np.array([[state, opp_action]]))),
            [2, -1]
        )

    def get_opponent_model_params(self, state):
        return tf.reshape(
            self.opponent_model(self._tfarray(np.array([[state]]))),
            [2, -1]
        )

    def compute_marginal_policy(self, state):
        # print(state)
        opp_model_mu, opp_model_sigma = self.get_opponent_model_params(state)
        sampled_opp_action = np.random.normal() * opp_model_sigma + opp_model_mu
        policy_mu, policy_sigma = self.get_policy_params(state, sampled_opp_action)
        # calculate the distribution of product of two dist.
        mu1, mu2, sig1, sig2 = opp_model_mu, policy_mu, opp_model_sigma, policy_sigma
        num1 = tf.multiply(mu1, tf.pow(sig2, 2))
        num2 = tf.multiply(mu2, tf.pow(sig1, 2))
        num = tf.add(num1, num2)
        denom = tf.add(tf.pow(sig2, 2), tf.pow(sig1, 2))
        mu = tf.divide(num, denom)
        num = tf.multiply(tf.pow(sig1, 2), tf.pow(sig2, 2))
        denom = tf.add(tf.pow(sig1, 2), tf.pow(sig2, 2))
        sigma = tf.divide(num, denom)
        return mu, sigma


    def act(self, state):
        mu, sigma = self.compute_marginal_policy(state)
        return np.random.normal() * sigma + mu

    def save_history(self, event):
        self.buffer[-1].append(event)

    def start_new_batch(self):
        self.buffer.append([])

    def update_P(self, window_length):
        window_length = min(len(self.buffer), window_length)
        m = defaultdict(list)
        for mini_batch in self.buffer[-window_length:]:
            for s, _, a_opp, _, _ in mini_batch:
                m[s].append(a_opp)
        for s in m:
            self.P_s[s] = [np.mean(m[s]), np.std(m[s])]

    def update_params(self):
        batch = self.buffer[-1]
        batch_len = tf.constant(len(batch))
        loss_1 = tf.constant(0., dtype=tf.double)
        ptr = tf.constant(0)

        # wrt theta
        def cond1(ptr, loss):
            return tf.less(ptr, batch_len)

        def body1(ptr, loss):
            s, a, a_opp, _, r = batch[ptr]
            ptr = tf.add(ptr, 1)
            marginal = tf.distributions.Normal(*self.compute_marginal_policy(s))
            opponent = tf.distributions.Normal(*self.get_opponent_model_params(s))
            conditional = tf.distributions.Normal(*self.get_policy_params(s, a_opp))
            phi_div_tsi = tf.divide(
                opponent.prob(a_opp),
                tf.distributions.Normal(*self.P_s[s]).prob(a_opp)
            )
            mult_1 = tf.multiply(
                phi_div_tsi,
                conditional.prob(a)
            )
            first = tf.multiply(mult_1, r)
            second_1 = tf.log(marginal.prob(a))
            second_2 = tf.log(marginal.prob(a))
            second_2 = tf.stop_gradient(second_2)
            second_2 = tf.add(second_2, 1)
            second = tf.multiply(second_1, second_2)

            loss = tf.add(loss, tf.add(first, second))
            return ptr, loss

        loss_1_func = partial(
            tf.divide,
            tf.while_loop(cond1, body1, [ptr, loss_1])[1],
            float(batch_len)
        )

        theta_variables = self.policy.get_trainable()
        # gradients = tf.gradients(loss_1, theta_variables)
        tf.train.AdamOptimizer(
            learning_rate=self.alpha).minimize(loss_1_func, var_list=theta_variables)

        # wrt phi
        ptr = tf.constant(0)
        loss_2 = tf.constant(0., dtype=tf.double)
        def cond2(ptr, loss):
            return tf.less(ptr, batch_len)

        def body2(ptr, loss):
            s, a, a_opp, _, r = batch[ptr]
            ptr = tf.add(ptr, 1)
            opponent = tf.distributions.Normal(*self.get_opponent_model_params(s))
            conditional = tf.distributions.Normal(*self.get_policy_params(s, a_opp))
            first_1 = tf.divide(
                opponent.prob(a_opp),
                tf.distributions.Normal(*self.P_s[s]).prob(a_opp)
            )
            first_1 = tf.multiply(first_1, opponent.prob(a_opp))
            first_2 = tf.add(
                r, tf.multiply(-conditional.prob(a), tf.log(conditional.prob(a)))
            )
            first = tf.multiply(first_1, first_2)
            second = tf.distributions.kl_divergence(
                opponent,
                tf.distributions.Normal(*self.P_s[s])
            )
            loss = tf.add(loss, tf.subtract(first, second))
            return ptr, loss

        loss_2_func = partial(
            tf.divide, 
            tf.while_loop(cond2, body2, [ptr, loss_2])[1],
            float(batch_len)
        )
        phi_variables = self.opponent_model.get_trainable()
        tf.train.AdamOptimizer(
            learning_rate=self.beta).minimize(loss_2_func, var_list=phi_variables)



        
        
