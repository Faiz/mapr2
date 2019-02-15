from pgagent import PGAgent
from maci.environments.differential_game import DifferentialGame
import numpy as np
import tensorflow as tf

GAME_NAME = "ma_softq"
AGENT_NUM = 2
MOVING_WINDOW_LEN = 5 # 5 mini batches => 5 * T, 500 games.

def play_differential_game(alpha=0.001, beta=0.001, discount=0.9, num_agents=2, episodes=100, iteration=1000):
    agents = []
    env = DifferentialGame(game_name=GAME_NAME, agent_num=AGENT_NUM)
    for i in range(num_agents):
        agents.append(PGAgent(i))
    device_config = tf.ConfigProto()
    with tf.Session(config=device_config) as sess:
        for _ in range(iteration):
            _ = [agent.start_new_batch() for agent in agents]
            for _ in range(episodes):
                states = env.reset()
                actions = np.array([
                    agent.act(state) for state, agent in zip(states, agents)
                ])
                # print(actions)
                state_primes, rewards, _, _ = env.step(actions)
                print(rewards)
                for agent_id, agent in enumerate(agents):
                    agent.save_history(
                        [
                            tf.reshape(states[agent_id], [-1, 1]),
                            actions[agent_id],
                            actions[1 - agent_id],
                            state_primes[agent_id],
                            rewards[agent_id],
                        ]
                    )
            # update P-tsi for each agent.
            _ = [agent.update_P(MOVING_WINDOW_LEN) for agent in agents]
            # update the parameters.
            _ = [agent.update_params() for agent in agents]

if __name__ == "__main__":
    # tf.enable_eager_execution()
    play_differential_game()