# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function

import sys
import random
import getopt
import resource
import psutil
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from hanabi_learning_environment.agents.greedy_agent import GreedyAgent
from hanabi_learning_environment.agents.q_agent import QAgent
from hanabi_learning_environment.agents.q_util import get_next_state
from plotter import plot_hist, plot_learning_rewards, plot_actions

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'QAgent': QAgent}

def get_memory_usage(obj):
  size = sys.getsizeof(obj)
  if 1024 <= size < 1024**2:
    return size // 1024, 'k'
  if 1024**2 <= size < 1024**3:
    return size // (1024**2), 'M'
  if 1024**3 <= size:
      return size // (1024**3), 'G'
  return size, ''

def print_memory_usage(agents, runner):
  for agent_id, agent in enumerate(agents):
    size, multiplier = get_memory_usage(agent)
    print('Size of agent {} state dict in {}B: {}'.format(agent_id, multiplier, size))
  size, multiplier = get_memory_usage(runner)
  print('Size of runner: {} {}B'.format(size, multiplier))
  print('Free RAM: {}'.format(psutil.virtual_memory().available))
  print('RAM used by application: {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

class Runner():
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players'], 'alpha': flags['alpha'],
                         'gamma': flags['gamma']}
    self.environment = rl_env.make('Hanabi-My-Small', num_players=flags['players'],
                                   seed=flags['seed'])
    self.agent_class = AGENT_CLASSES[flags['agent_class']]

  def play_episode(self, agents, actions):
    observations = self.environment.reset()
    done = False
    episode_reward = 0
    agents_rewards = {i: 0 for i, _ in enumerate(agents)}
    while not done:
      for agent_id, agent in enumerate(agents):
        observation = observations['player_observations'][agent_id]
        def get_next_state_wrapper(action):
          return get_next_state(self.environment.state.copy(), agent_id, action,
                                self.environment.game)
        action, reward = agent.act(observation, get_next_state_wrapper)
        if observation['current_player'] == agent_id:
          assert action is not None
          current_player_action = action
          actions[agent_id][action['action_type']] += 1
          agents_rewards[agent_id] += reward
        else:
          assert action is None
      # Make an environment step.
      # Done only once, when current action != None
      observations, _, done, _ = self.environment.step(current_player_action)
    episode_reward = self.environment.state.score() #reward
    print('Episode reward: {}'.format(episode_reward))
    return episode_reward, agents_rewards, actions

  def run(self, num_episodes, agents=None):
    """Run episodes."""
    rewards = []
    if agents is None:
      agents = [self.agent_class(self.agent_config)
                for _ in range(self.flags['players'])]
    agents_rewards = {i: [] for i, _ in enumerate(agents)}
    actions = {i: {'DISCARD': 0, 'PLAY': 0, 'REVEAL_COLOR': 0, 'REVEAL_RANK': 0}
               for i, _ in enumerate(agents)}
    for episode in range(num_episodes):
      print('=' * 100)
      print('Episode: {}'.format(episode))
      episode_reward, agent_ep_rewards, actions = self.play_episode(agents, actions)
      rewards.append(episode_reward)
      for agent_id, _ in enumerate(agents):
        agents_rewards[agent_id].append(agent_ep_rewards[agent_id])
      print('Max reward: %.3f' % max(rewards))
      print_memory_usage(agents, self)
      print('=' * 100)
    print('Final max reward: %.3f' % max(rewards))
    return agents, rewards, agents_rewards, actions

def main():
  flags = {'players': 2, 'num_episodes': 1, 'num_test_episodes': 1, 'agent_class': 'SimpleAgent',
           'alpha': 0.1, 'gamma': 0.9, 'seed': 12345}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'num_test_episodes=',
                                      'agent_class=',
                                      'seed=',
                                      'alpha=',
                                      'gamma='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--num_test_episodes  number of test game episodes to run.\n'
             '--alpha         step size for Q-learning. \n'
             '--gamma         discount rate for Q-learning. \n'
             '--seed          random generator seed. \n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)

  random.seed(flags['seed'])

  form_str = 'players_{}-episodes_{}-test_episodes_{}-agent_{}-alpha_{}-gamma_{}-seed_{}'
  suffix = form_str.format(flags['players'], flags['num_episodes'], flags['num_test_episodes'],
                           flags['agent_class'], flags['alpha'], flags['gamma'], flags['seed'])
  train_suffix = suffix + '-train'
  test_suffix = suffix + '-test'

  runner = Runner(flags)

  print('*' * 100)
  print('TRAINING')
  agents, rewards, agents_rewards, actions = runner.run(flags['num_episodes'])
  print('*' * 100)

  player_num = len(agents)

  for ar in agents_rewards:
    plot_learning_rewards(agents_rewards[ar], ar, train_suffix)
    plot_actions(actions[ar], ar, train_suffix)
  plot_hist(rewards, -1, train_suffix, flags['players'])

  if flags['agent_class'] == 'QAgent':
    greedy_agents = [GreedyAgent(runner.agent_config, agent.Q) for agent in agents]
    print('*' * 100)
    print('TESTING')
    _, rewards, agents_rewards, actions = runner.run(flags['num_test_episodes'], greedy_agents)
    print('*' * 100)
    plot_hist(rewards, -1, test_suffix, flags['players'])
    for ar in agents_rewards:
      plot_actions(actions[ar], ar, test_suffix)

  print('rewards:\n{}'.format(rewards))
  print('agent 0 rewards:\n{}'.format(agents_rewards[0]))
  print('agent 1 rewards:\n{}'.format(agents_rewards[1]))

if __name__ == "__main__":
  main()
