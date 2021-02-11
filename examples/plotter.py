# pylint: disable=missing-function-docstring, invalid-name, too-many-arguments
"""Plotting utilities."""

from statistics import mean, median, stdev
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# We show histogram for max reward per game rules (number of fireworks cards)
MAX_REWARD=12

def bins_labels(bins, sparsity, **kwargs):
  n = len(bins)
  max_b = max(bins)
  min_b = min(bins)
  bin_w = (max_b - min_b) / (n - 1)
  bin_ticks = np.arange(min_b+bin_w/2, max_b + bin_w, sparsity*bin_w)
  sparse_bins = np.take(bins, range(0, max(bins), sparsity))
  plt.xticks(bin_ticks, sparse_bins, **kwargs)
  plt.close()

def plot_hist(rewards, agent_id, suffix, player_number, save_csv=True):
  fig, ax = plt.subplots(1, 1)
  bins = range(MAX_REWARD + 1)
  ax.hist(rewards, bins=bins, density=True)
  bins_labels(bins, sparsity=5)
  ax.set_xlabel('Points')
  ax.set_ylabel('Proportion of games')

  mean_reward = mean(rewards)
  median_reward = median(rewards)
  stdev_reward = stdev(rewards)
  n = len(rewards)

  form_str = '{} players\nMean score = {:.2f}\nMedian score = {:.0f}\ns.d. = {:.2f}\nn = {}'
  ax.text(0.98, 0.98, form_str.format(player_number, mean_reward, median_reward, stdev_reward, n),
          ha='right', va='top', transform=plt.gca().transAxes)

  fig.tight_layout()
  filename = 'hist_rewards-player_{}-{}'.format(agent_id, suffix)
  fig.savefig('{}.png'.format(filename), bbox_inches='tight')
  if save_csv:
    np.savetxt('{}.csv'.format(filename), np.array(rewards), delimiter=',')
  plt.close()

def plot_learning_rewards(rewards, agent_id, suffix, save_csv=True):
  fig, ax = plt.subplots(1, 1)
  ax.plot(range(len(rewards)), rewards, '.')
  ax.set_xlabel('Episodes')
  ax.set_ylabel('Sum of rewards during episode')
  ax.get_yaxis().set_major_formatter(
      matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
  fig.tight_layout()
  filename = 'learning_rewards-player_{}-{}'.format(agent_id, suffix)
  fig.savefig('{}.png'.format(filename), bbox_inches='tight')
  if save_csv:
    np.savetxt('{}.csv'.format(filename), np.array(rewards, dtype=int), delimiter=',', fmt='%i')
  fig.clear()
  plt.close(fig)

def plot_actions(actions, agent_id, suffix, save_csv=True):
  fig, ax = plt.subplots(1, 1)
  keys = list(actions.keys())
  vals = list(actions.values())
  ax.bar(keys, vals)
  ax.set_xlabel('Actions')
  ax.set_ylabel('Count')

  fig.tight_layout()
  filename = 'actions-player_{}-{}'.format(agent_id, suffix)
  fig.savefig('{}.png'.format(filename), bbox_inches='tight')
  header = ','.join(k for k in keys)
  if save_csv:
    np.savetxt('{}.csv'.format(filename), vals, header=header, delimiter=',')
  plt.close()
