

import numpy as np
from matplotlib import pyplot as plt

from world import World
from utils import  getpolicy, getvalue, plotarrows

plt.rcParams['figure.facecolor']='white'


# Initialization
# Initialize the world, Q-table, and hyperparameters
world1 = World(1)
q_table = np.zeros((world1.y_size, world1.x_size,4))
#q_table.hstack()
learning_rate = 0.01
discount = 0.99
episodes = 5
epsilon = 0.9
start_epsilon_decaying = 1
end_epsilon_decaying = episodes//2
epsilon_decay_value = epsilon/(end_epsilon_decaying - start_epsilon_decaying)


# Training loop
# Train the agent using the Q-learning algorithm.

# act 1 -> bottom
# act 2 -> top
# act 3 -> right
# act 4 -> left

import random
for episode in range(episodes):

    world = World(1)

    done = False
    current_state = world.pos
    print(current_state)
    #print(world1.draw())
    while not done:

        act = getvalue(q_table[current_state])

        if current_state[1] == 14:
            act = random.choice([2,4])
        if current_state[0] == 9:
            act = random.choice([2,3])
        if current_state[0] == 0:
            act = random.choice([1,3])
        if current_state[1] == 0:
            act = random.choice([2,3])
        if current_state[0] == 0 and current_state[1] ==14:
            act = random.choice([1,4])
        if current_state[0] == 9 and current_state[1] ==0:
            act = random.choice([2,3])
        if current_state[0] == 0 and current_state[1] ==0:
            act = random.choice([1,3])
        if current_state[0] == 9 and current_state[1] ==14:
            act = random.choice([2,4])
        print(act)
        value, reward = world.action(act)

        print(value,reward)
        new_state = world.pos
        #plotarrows(new_state)
        print(new_state)
        if not done:
            max_future_q = getpolicy(q_table[new_state])
            current_q = getpolicy(q_table[current_state + (act-1,)])


            new_q = (1-learning_rate) * current_q + learning_rate*(reward + discount*max_future_q)
            q_table[current_state + (act-1,)] = new_q


        elif current_state == world.term:
            q_table[current_state + (act-1,)] = 0
            done= True
            print(f'We have made it to episode {episode}')

        current_state = new_state

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value
