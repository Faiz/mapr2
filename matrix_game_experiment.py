from maci.environments import MatrixGame
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt

AGENT_NUM = 2
ACTION_NUM = 2
GAME_NAME = "matching_pennies"
ITERATION = 1000
SAMPLE_SIZE = 1
K = 1


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
            agent.update_policy(sample_size=SAMPLE_SIZE, k=K)
    print(reward_history)
    
    
