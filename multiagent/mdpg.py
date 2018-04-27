import tensorflow as tf
import numpy as np
from multiagent.actor_network import ActorNetwork
from multiagent.critic_network import CriticNetwork
import matplotlib.pyplot as plt
from common.grid_world import GridWorld
from multiagent.replay_buffer import ReplayMemory

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

RENDER = False
# env = gym.make(MultiAgentEnv)

###############################  DDPG  ####################################


class MDPG(object):
    def __init__(self, gamma=0.0975, epsilon=1, epsilon_decay=0.99, terminal_reward=10, env=None,
                 memory=None, reward_discount_factor=0.0, model_name=None, total_cars=2, grid_size=6, learning_rate=0.5):
        self.env = env
        self.losses = []
        self.all_games_hisotry = []
        self.replay = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.reward_discount_factor = reward_discount_factor
        self.terminal_reward = terminal_reward
        self.grid_size = grid_size
        self.total_cars = total_cars
        # self.model = self.loadModel(model_name)
        self.learning_rate = learning_rate
        self.action_dim = 4

        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(sess)
        self.actor = ActorNetwork(sess, grid_size*grid_size, 4, BATCH_SIZE, TAU, LR_A)
        self.critic = CriticNetwork(sess, grid_size*grid_size, 4, BATCH_SIZE, TAU, LR_C)

    def createTraining(self, minibatch):
        x_train = []
        y_train = []
        for memory in minibatch:
            old_state_m, action_m, reward_m, new_state_m, terminal_m = memory
            old_qval = self.model.predict(np.array((old_state_m,)), batch_size=1)
            newQ = self.model.predict(np.array((new_state_m,)), batch_size=1)
            maxQ = np.max(newQ)

            y = np.zeros((1, self.env.num_actions))
            y[:] = old_qval[:]

            if not terminal_m:
                update = (reward_m + (self.gamma * maxQ))
            else:
                update = reward_m

            y[0][action_m] = update

            x_train.append(old_state_m)
            y_train.append(y.reshape(self.env.num_actions,))

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train, y_train

    def graphLoss(self):
        epochs = []
        losses_A = []
        count = 0
        for y in self.losses:
            losses_A.append(y[0])
            epochs.append(count)
            count = count + 1
        plt.plot(epochs[:], losses_A[:])

    def train(self, epochs, episodes, max_episode_length, output_weights_name):
        for i in range(epochs+1):
            total_reward = 0
            for j in range(episodes):
                loss = 0
                terminal = False
                self.env.reset()
                while not terminal:
                    self.env.t += 1
                    print("Epoch #: %s" % (i,))
                    print("Game #: %s" % (j,))
                    print("Time step #: %s" % (self.env.t,))
                    print("Epsilon :", float(self.epsilon))
                    # self.env.print_grid()

                    curr_history = {}
                    curr_history['t'] = self.env.t
                    curr_history['curr_state'] = (self.env.cars_grid.copy(), self.env.cust_grid.copy())
                    all_agents_step = self.env.stepAll(self.critic, self.epsilon)
                    curr_history['actions'] = all_agents_step
                    curr_history['next_state'] = (self.env.cars_grid.copy(), self.env.cust_grid.copy())
                    self.env.history.append(curr_history)

                    terminal = self.env.isTerminal()
                    for memory in all_agents_step:
                        self.replay.addToMemory(memory, terminal)
                if self.replay.isFull():
                    minibatch = self.replay.getMinibatch()
                    # states = np.asarray(e[0] for e in minibatch)
                    # actions = np.asarray(e[1] for e in minibatch)
                    # rewards = np.asarray(e[2] for e in minibatch)
                    # new_states = np.asarray(e[3] for e in minibatch)
                    # dones = np.asarray(e[4] for e in minibatch)
                    # y_t = np.asarray(e[1] for e in minibatch)
                    states, actions, rewards, new_states, dones = minibatch[0]
                    y_t = actions

                    # x_train, y_train = self.createTraining(minibatch)

                    # a_t_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                    # a_t = self.env.intended_step(model=self.actor, epsilon=self.epsilon)
                    # s_t1, r_t, done, info = self.env.step(a_t)

                    # learn
                    # print(str(new_states.size()))
                    target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

                    for k in range(len(minibatch)):
                        if dones[k]:
                            y_t[k] = rewards[k]
                        else:
                            y_t[k] = rewards[k] + GAMMA*target_q_values[k]

                    loss += self.critic.model.train_on_batch([states, actions], y_t)
                    a_for_grad = self.actor.model.predict(states)
                    grads = self.critic.gradients(states, a_for_grad)
                    self.actor.train(states, grads)
                    self.actor.target_train()
                    self.critic.target_train()

                    # total_reward += r_t
                    # s_t = s_t1


if __name__ == '__main__':
    env = GridWorld()
    memory = ReplayMemory(buffer=50000, batchSize=500)
    mdpg = MDPG(memory=memory, env=env)
    mdpg.train(100, 100, 100, 'wights')




