import numpy as np
from copy import deepcopy
import argparse
import torch

from mddpg.Mddpg import MDDPG
from common.util import *
from common.grid_world import GridWorld
from mddpg.evaluator import Evaluator
import os
ROOT = os.path.dirname(os.path.realpath(__file__))
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

def train(num_iterations, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0
    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action
        for i in range(args.n_agents):
            if step <= args.warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(observation)

        # env response with next_observation, reward, done
        obs_next, reward, done, info = env.step(action)
        obs_next = deepcopy(obs_next)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, obs_next, done)
        if step > args.warmup:
            agent.update_policy()

        # evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug:
                prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # save intermediate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        #update
        step += 1
        episode_steps += 1
        for i in reward:
            episode_reward += i
        observation = deepcopy(obs_next)

        if done:
            if debug:
                prGreen('#{}: episode_reward:{} step:{}'.format(episode, episode_reward, step))
            agent.memory.add(
                observation,
                agent.select_action(observation),
                reward,
                obs_next,
                False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0
            episode += 1


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug:
            prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent DDPG')
    # add argument
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    # parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=600, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--Crate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--Arate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default=ROOT+'/output', type=str, help='')
    parser.add_argument('--debug', default=True, dest='debug', action='store_true')
    # parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # new added
    parser.add_argument('--max_episode_len', default=400, type=int, help='maxepisode length')
    parser.add_argument('--n_agents', default=2, type=int, help='number of agents in the environment')
    parser.add_argument('--grid_size', default=6, type=int)

    # parser args
    args = parser.parse_args()
    # set env
    env = GridWorld(args=args)
    n_obs = env.observation_shape
    n_actions = env.action_shape
    evaluate = Evaluator(args.validate_episodes, args.validate_steps, args.output,
                         max_episode_length=args.max_episode_length)
    agent = MDDPG(n_obs=n_obs, n_actions=n_actions, args=args, n_states=n_obs)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(env)

    if args.mode == 'train':
        train(args.train_iter, env, evaluate, args.validate_steps, args.output,
              max_episode_length=args.max_episode_length, debug=args.debug)
    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)