import torch.nn as nn
import numpy as np
from torch.optim import Adam
from mddpg.model import Actor, Critic
from common.util import *
from multiagent.replay_memory import ReplayMemory
from mddpg.random_process import OrnsteinUhlenbeckProcess

# GLOBAL
criterion = nn.MSELoss()


class MDDPG(object):
    def __init__(self, n_states, n_obs, n_actions, args):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_obs = n_obs
        # config
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
        }
        # build the Actor network
        self.actor = Actor(self.n_obs, self.n_actions, **net_cfg)
        self.actor_target = Actor(self.n_obs, self.n_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.Arate)
        # build the Critic network
        self.critic = Critic(self.n_obs, self.n_actions, **net_cfg)
        self.critic_target = Critic(self.n_obs, self.n_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.Crate)
        # Update the target network
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.d_epsilon = 1.0 / args.epsilon

        # Others
        self.n_agents = args.n_agents
        self.grid_size = args.grid_size

        # Other parameters
        self.epsilon = 1.0
        self.s_t = np.zeros(self.n_agents)  # Most recent state
        self.a_t = np.zeros(self.n_agents)  # most recent action
        self.is_training = None

        # Create replay buffer
        self.memory = ReplayMemory(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        self.random_process = OrnsteinUhlenbeckProcess(theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma,
                                                       size=n_actions)

        # Using CUDA
        if USE_CUDA:
            self.cuda()

    def update_policy(self):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.memory.sample(self.batch_size)
        # Calculate the q values
        for i in range(self.n_agents):
            # reshape next_obs_batch to 64 * 4 * 5 * 5
            next_obs_batch_reshape = np.squeeze(next_obs_batch[:, i, :, :, :])
            # reshape action_batch
            next_q_values = self.critic_target([to_tensor(next_obs_batch_reshape, volatile=True),
                                               self.actor_target(to_tensor(next_obs_batch_reshape, volatile=True))])
            next_q_values.volatile = False
            mat = to_tensor(done_batch[i].astype(np.float).reshape(-1, 1))
            target_q_batch = to_tensor(reward_batch[:, i].reshape(-1, 1)) + \
                self.discount * mat * next_q_values
            # Critic update
            self.critic.zero_grad()
            obs_batch_reshape = np.squeeze(obs_batch[:, i, :, :, :])
            q_batch = self.critic([to_tensor(obs_batch_reshape, volatile=True),
                                   self.actor_target(to_tensor(obs_batch_reshape, volatile=True))])
            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optim.step()
            prPurple('critic loss: {}'.format(to_numpy(value_loss)[0]))

            # Actor update
            self.actor.zero_grad()
            policy_loss = -self.critic(to_tensor(obs_batch.reshape(-1, 1000)))

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()
            prRed('actor loss: {}'.format(to_numpy(policy_loss)[0]))

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def random_action(self):
        action = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            action[i] = np.random.choice(self.n_actions, 1)
            self.a_t = action
        return action

    def select_action(self, obs_t, decay_epsilon=True):
        action = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            matrix = to_numpy(self.actor(to_tensor(np.array(obs_t[i].reshape(1,4,6,6)))))
            matrix = np.reshape(matrix, (4, -1))
            # print(matrix)
            action[i] = np.argmax(np.sum(matrix, axis=1))
            # print(action[i])
            # action[i] += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
            # action[i] = np.clip(action[i], -1., 1.)
            # Decay epsilon
            if decay_epsilon:
                self.epsilon -= self.d_epsilon
            self.a_t[i] = action[i]
        return action

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.add(self.s_t, self.a_t, r_t, s_t1, done)
            self.s_t = s_t1

    def reset(self, obs):
        self.s_t = obs

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def load_weights(self, input):
        if input is None:
            return
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(input))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(input))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    @staticmethod
    def seed(s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
