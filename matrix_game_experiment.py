from maci.environments import MatrixGame
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt

AGENT_NUM = 2
ACTION_NUM = 2
GAME_NAME = "wolf_05_05"
ITERATION = 5000
SAMPLE_SIZE = 1
K = 10


if __name__ == "__main__":
    env = MatrixGame(
        game=GAME_NAME,
        agent_num=AGENT_NUM,
        action_num=ACTION_NUM,
    )

    agents = []
    for i in range(AGENT_NUM):
        agent = Agent(id, ACTION_NUM, env)
        agents.append(agent)
    
    reward_history = []
    for _ in range(ITERATION):
        # reset resets the game and returns [0, 0]
        # since in matrix game, you don't really have a 
        # state or think of it as start state.
        for _ in range(SAMPLE_SIZE):
            states = env.reset()
            actions = np.array([
                agent.act(state)[0] for state, agent in zip(states, agents)
            ])
            state_primes, rewards, _, _ = env.step(actions)
            reward_history.append(rewards)
            for agent_index, (state, reward, state_prime, agent) in enumerate(
                zip(states, rewards, state_primes, agents)):
                agent.update_opponent_action_prob(
                    state,
                    actions[agent_index],
                    actions[1 - agent_index],
                    state_prime,
                    reward,
                )
        # Update Q
        for agent in agents:
            agent.update_policy(sample_size=SAMPLE_SIZE, k=K)
    history_pi_0 = [p[1] for p in agents[0].pi_history]
    history_pi_1 = [p[1] for p in agents[1].pi_history]

    cmap = plt.get_cmap('viridis')
    colors = range(len(history_pi_1))
    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(211)

    scatter = ax.scatter(history_pi_0, history_pi_1, c=colors, s=1)
    ax.scatter(0.5, 0.5, c='r', s=10., marker='*')
    colorbar = fig.colorbar(scatter, ax=ax)

    ax.set_ylabel("Policy of Player 2")
    ax.set_xlabel("Policy of Player 1")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # ax = fig.add_subplot(212)

    # ax.plot(history_pi_0)
    # ax.plot(history_pi_1)

    plt.tight_layout()
    plt.show()