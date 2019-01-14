import numpy as np
from collections import defaultdict, Counter
from functools import partial
from typing import List

class Agent:
    def __init__(self, id, action_num, env):
        self.id = id
        self.alpha_decay_steps = 10000.
        self.epoch = 0
        self.action_num = action_num
        self.env = env
        # Policy for the agent
        self.pi = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )
        self.Q = defaultdict(
            partial(np.random.rand, *(self.action_num, action_num))
        )
        self.pi_neg_i = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )
        # (state, agent_action, opp_action, state_prime, reward)
        self.replay_buffer = []
        self.pi_history = []
        self.cond_pi = defaultdict(
            partial(np.random.rand, *(self.action_num, action_num))
        )
        self.rho = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )
        self.marginal_pi = defaultdict(
            partial(np.random.dirichlet, [1.0] * self.action_num)
        )

    def update_opponent_action_prob(self, s, a_i, a_neg_i, s_prime, r):
        self.replay_buffer.append((s, a_i, a_neg_i, s_prime, r))

        a_neg_i_map = Counter()
        # update opponent action probability
        denominator = 0
        for exp in self.replay_buffer:
            if exp[0] != s: # state
                continue
            denominator += 1
            a_neg_i_map[exp[2]] += 1
        self.pi_neg_i[s] = np.array(
            [a_neg_i_map[i] / denominator for i in range(self.action_num)])

    def step_decay(self):
        return self.alpha_decay_steps / (self.alpha_decay_steps + self.epoch)

    def compute_conditional_pi(self, s):
        self.cond_pi[s] = np.exp(self.Q[s])
        self.cond_pi[s] /= np.sum(np.exp(self.Q[s]), axis=0)
        return self.cond_pi[s]

    def compute_opponent_model(self, s):
        self.rho[s] = np.multiply(
            self.pi_neg_i[s],
            np.sum(np.exp(self.Q[s]), axis=0)
        )
        self.rho[s] /= self.rho[s].sum()
        return self.rho[s]

    def compute_marginal_pi(self, s):
        pi = self.compute_conditional_pi(s)
        rho = self.compute_opponent_model(s)
        self.marginal_pi[s] = np.sum(np.multiply(pi, rho), 1)
        return self.marginal_pi[s]

    def update_policy(self, sample_size, k, sliding_wnd_size=10, gamma=0.95):
        sliding_wnd_size = min(sliding_wnd_size, len(self.replay_buffer))
        sliding_window = self.replay_buffer[-sliding_wnd_size:]
        samples = np.random.choice(len(sliding_window), size=sample_size)
        decay_alpha = self.step_decay()
        for exp in samples:
            s, a_i, a_neg_i, s_prime, r = sliding_window[exp]
            numerator, denominator = 0, 0
            for _ in range(k):
                sampled_a_i, sampled_a_neg_i = self.act(s)
                numerator += (
                    self.pi_neg_i[s_prime][sampled_a_neg_i] *
                    np.exp(self.Q[s_prime][sampled_a_i, sampled_a_neg_i])
                )
                pi = self.compute_conditional_pi(s_prime)[sampled_a_i, sampled_a_neg_i]
                rho = self.compute_opponent_model(s_prime)[sampled_a_neg_i]
                denominator += (pi * rho)
            
            v_s_prime =  np.log((1 / k) * (numerator / denominator))
            y = r + gamma * v_s_prime
            self.Q[s][a_i, a_neg_i] = (
                (1 - decay_alpha) * self.Q[s][a_i, a_neg_i] +
                decay_alpha * y
            )
        self.epoch += 1
        self.pi_history.append(self.compute_marginal_pi(s))
        
    def act(self, s):
        """
        Function act sample actions from pi for a given state.
        Input: 
            s: Int representing a state.
        Returns:
            Int: Sampled action for agent i, 
                 Sampled action for the opponent according to our
                 belief.
        """
        opponent_p = self.compute_opponent_model(s)
        # print(opponent_p)
        opponent_action = np.random.choice(
            opponent_p.size, size=1, p=opponent_p)[0]
        agent_p = np.exp(self.Q[s][:, opponent_action])

        agent_action = np.random.choice(
            agent_p.size, size=1, p=agent_p/agent_p.sum())[0]
        return agent_action, opponent_action
