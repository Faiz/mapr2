import tensorflow as tf
import numpy as np
from collections import defaultdict
from functools import partial
from policy import NN
from dist import DiagGaussianPd
import tf_utils as U
from maci.misc import tf_utils


class PGAgent:
    def __init__(self, id, ob_space_shape=(1,), policy_output_num=2, opponent_model_output_num=2,
                 action_range=10., alpha=0.001, beta=0.001):
        self.id = id
        self.con_policy_network = NN(policy_output_num)
        self.opponent_model_network = NN(opponent_model_output_num)
        self.buffer = []
        self.alpha = alpha
        self.beta = beta
        self.action_range = action_range

        self.P_s = None
        self.mar_pd = None
        self.con_pd = None
        self.opp_pd = None
        self.predicted_opp_ac = None
        sequence_length = None
        self.theta_ops = None
        self.phi_ops = None

        self.tf_state = U.get_placeholder(name="ctrl_ob", dtype=tf.float64, shape=[sequence_length] + list(ob_space_shape))
        _ = self.compute_marginal_policy(self.tf_state)
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.tf_ret = tf.placeholder(dtype=tf.float64, shape=[None], name='tf_ret')  # Empirical return
        self.tf_ac = self.mar_pd.pdtype.sample_placeholder([None], name='tf_ac')  # action
        self.tf_opp_ac = self.mar_pd.pdtype.sample_placeholder([None], name='tf_ac')  # action of opponents

        ac = U.switch(self.stochastic, self.mar_pd.sample(), self.mar_pd.mode())
        self._act = U.function([self.stochastic, self.tf_state], [ac])
        self.set_learning_ops()
        self._training_ops = [self.theta_ops, self.phi_ops]

        self._sess = tf_utils.get_default_session()
        self._sess.run(tf.global_variables_initializer())


    def _tfarray(self, arr):
        return tf.convert_to_tensor(arr)

    def get_policy_dist(self, state, opp_action):
        input = tf.concat([state, opp_action], axis=1)
        # input = tf.reshape(tf.stack([state, opp_action], axis=1), [-1, 2])
        flat = self.con_policy_network(input)
        self.con_pd = DiagGaussianPd(flat)
        return self.con_pd

    def get_opponent_model_dist(self, state):
        flat = self.opponent_model_network(state)
        self.opp_pd = DiagGaussianPd(flat)
        return self.opp_pd

    def compute_marginal_policy(self, state):
        # state = tf.reshape(state, [-1, 1])
        flat_opp = self.get_opponent_model_dist(state)
        self.predicted_opp_ac = flat_opp.sample()
        flat_p = self.get_policy_dist(state, self.predicted_opp_ac)
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
        self.mar_pd = DiagGaussianPd(flat_m)
        return self.mar_pd

    def act(self, state, stochastic):
        action = self._act(stochastic, state)
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

    def get_loss(self, tf_ac, tf_opp_ac, tf_ret):

        phi_div_tsi = tf.divide(self.opp_pd.prob(tf_opp_ac), self.P_s.prob(tf_opp_ac))
        mult_1 = tf.multiply(phi_div_tsi, self.con_pd.logp(tf_ac))
        first = tf.multiply(mult_1, tf_ret)
        second_1 = self.mar_pd.logp(tf_ac)
        second_2 = self.mar_pd.logp(tf_ac)
        second_2 = tf.stop_gradient(second_2)
        second_2 = tf.add(second_2, 1)
        second = tf.multiply(second_1, second_2)
        loss_1 = -tf.reduce_mean(tf.add(first, second))

        first_1 = tf.multiply(phi_div_tsi, tf_ret)
        first_2 = tf.multiply(self.opp_pd.logp(self.predicted_opp_ac), self.con_pd.logp(tf_ac))
        first = tf.add(first_1, first_2)
        second = self.opp_pd.kl(self.P_s)
        loss_2 = -tf.reduce_mean(tf.add(first, second))
        return loss_1, loss_2

    def set_learning_ops(self):
        loss_for_theta, loss_for_phi = self.get_loss(self.tf_ac, self.tf_opp_ac, self.tf_ret)

        theta_variables = self.con_policy_network.get_trainable()
        self.theta_ops = tf.train.AdamOptimizer(
            learning_rate=self.alpha).minimize(loss_for_theta, var_list=theta_variables)

        phi_variables = self.opponent_model_network.get_trainable()
        self.phi_ops = tf.train.AdamOptimizer(
            learning_rate=self.beta).minimize(loss_for_phi, var_list=phi_variables)

    def _get_feed_dict(self, states, actions, opp_actions, rewards):
        """Construct a TensorFlow feed dictionary from a sample batch."""
        feeds = {
            self.tf_state:states,
            self.tf_ac:actions,
            self.tf_opp_ac:opp_actions,
            self.tf_ret:rewards
        }
        return feeds

    def update_params(self, states, actions, opp_actions, rewards):
        feed_dict = self._get_feed_dict(states, actions, opp_actions, rewards)
        self._sess.run(self._training_ops, feed_dict)



    # def loss_1(self, s, a, a_i, r):
    #     self.mar_pd = self.compute_marginal_policy(s)
    #     phi_div_tsi = tf.divide(self.opp_pd.prob(a_i), self.P_s.prob(a_i))
    #     mult_1 = tf.multiply(phi_div_tsi, self.con_pd.logp(a))
    #     first = tf.multiply(mult_1, r)
    #
    #     second_1 = self.mar_pd.logp(a)
    #     second_2 = self.mar_pd.logp(a)
    #     second_2 = tf.stop_gradient(second_2)
    #     second_2 = tf.add(second_2, 1)
    #     second = tf.multiply(second_1, second_2)
    #     return tf.reduce_mean(tf.add(first, second))
    #
    # def loss_2(self, s, a, a_i, r):
    #     self.mar_pd = self.compute_marginal_policy(s)
    #     first_1 = tf.divide(self.opp_pd.prob(a_i), self.P_s.prob(a_i))
    #     first_1 = tf.multiply(first_1, self.opp_pd.logp(a_i))
    #     first_2 = tf.add(r, -tf.multiply(tf.exp(self.con_pd.logp(a)), self.con_pd.logp(a)))
    #     first = tf.multiply(first_1, first_2)
    #     second = self.opp_pd.kl(self.P_s)
    #     return tf.reduce_mean(tf.subtract(first, second))





    # def update_params(self):
    #     batch = np.array(self.buffer[-1])
    #
    #     states = batch[:, 0].reshape(-1, 1)
    #     actions = batch[:, 1].reshape(-1, 1)
    #     opp_actions = batch[:, 2].reshape(-1, 1)
    #     rewards = batch[:, 4].reshape(-1, 1)
    #
    #     theta_variables = self.con_policy_network.get_trainable()
    #
    #     loss_1_func = partial(
    #         self.loss_1,
    #         states, actions, opp_actions, rewards
    #     )
    #
    #     tf.train.AdamOptimizer(
    #         learning_rate=self.alpha).minimize(loss_1_func, var_list=theta_variables)
    #     loss_2_func = partial(
    #         self.loss_2,
    #         states, actions, opp_actions, rewards
    #     )
    #     phi_variables = self.opponent_model_network.get_trainable()
    #     tf.train.AdamOptimizer(
    #         learning_rate=self.beta).minimize(loss_2_func, var_list=phi_variables)
