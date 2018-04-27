import random
import numpy as np


class Car(object):
    def __init__(self, carID, location, grid_size, reward_backprop_rate, reward_others_discount_rate):
        self.carID = carID
        self.location = location
        self.grid_size = grid_size
        self.reward_self_raw = []
        self.reward_self_backprop = []
        self.reward_others_raw = []
        self.reward_others_backprop = []
        self.reward_others_discount_rate = reward_others_discount_rate
        self.reward_backprop_rate = reward_backprop_rate
        self.step = {}
        self.num_pick = 10

    def __str__(self):
        return "C{0}".format(self.carID)

    def update_reward_self(self, t, reward_val):
        self.reward_self_raw, self.reward_self_backprop = self.update_reward(t, reward_val, self.reward_self_raw,
                                                                             self.reward_self_backprop)

    def update_reward_others(self, t, reward_val):
        reward_val = reward_val * self.reward_others_discount_rate
        self.reward_others_raw, self.reward_others_backprop = self.update_reward(t, reward_val, self.reward_others_raw,
                                                                                 self.reward_others_backprop)

    def update_reward(self, t, reward_val, reward_raw, reward_backprob):
        if len(reward_raw) == 0:
            reward_raw.append((t, reward_val))
            reward_backprob.append((t, reward_val))
        elif reward_raw[-1][0] == t:
            reward_raw[-1] = (t, reward_raw[-1][1] + reward_val)
            reward_backprob[-1] = (t, reward_backprob[-1][1] + reward_val)
        else:
            reward_raw.append((t, reward_val))
            reward_backprob.append((t, reward_val))
        reward_backprob = self.update_backprop_reward(reward_backprob, reward_val)
        return reward_raw, reward_backprob

    def update_backprop_reward(self, list_to_backprop, reward_val):
        # backpropagate reward
        for i in range(len(list_to_backprop) - 1)[::-1]:
            reward_val = reward_val * self.reward_backprop_rate
            list_to_backprop[i] = (list_to_backprop[i][0], list_to_backprop[i][1] + reward_val)
        return list_to_backprop

    def choose_action(self, state, model, epsilon, num_actions, observation_shape, demand=None):
        if (random.random() < epsilon):
            action = np.random.randint(0, num_actions)
        else:
            # action = (np.argmax(model.predict(state.reshape(1,observation_shape), batch_size=1)))
            action = (np.argmax(model.predict(np.array((state,)), batch_size=1)))
        return action