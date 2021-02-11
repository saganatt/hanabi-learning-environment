"""Standalone script to retrieve more wins statistics"""
import numpy as np

SEEDS=[283723, 12345, 39845, 23458, 98437]
DIR='new-alpha-0.1_gamma-0.9_life-3_info-8_colors-4_ranks-3_hand-4/'
SIMPLE_STR='hist_rewards-player_-1-players_2-episodes_1000-test_episodes_1000-' + \
           'agent_SimpleAgent-alpha_0.1-gamma_0.9-seed_{}-train.csv'
Q_STR='hist_rewards-player_-1-players_2-episodes_100000-test_episodes_1000-' + \
      'agent_QAgent-alpha_0.1-gamma_0.9-seed_{}-{}.csv'

stat_dict = {'simple': 0., 'q train': 0., 'q test': 0.}
stats = {'all means': stat_dict.copy(), 'all std dev': stat_dict.copy(),
         'wins means': stat_dict.copy(), 'wins means std dev': stat_dict.copy(),
         'wins count': stat_dict.copy(), 'wins percs': stat_dict.copy()}
for seed in SEEDS:
  files = {'simple': DIR + SIMPLE_STR.format(seed),
           'q train': DIR + Q_STR.format(seed, 'train'),
           'q test': DIR + Q_STR.format(seed, 'test')}
  print('seed: {}'.format(seed))
  for fkey in files:
    res = np.genfromtxt(files[fkey], delimiter=',')
    wins = res[res > 0.]

    num_all = res.shape[0]
    mean_all = np.mean(res)
    std_dev_all = np.std(res)

    num_wins = wins.shape[0]
    mean_win = np.mean(wins)
    std_dev_win = np.std(wins)
    perc_wins = 100. * num_wins / num_all

    print('{} num all: {} wins: {} perc: {:.3f}'.format(fkey, num_all, num_wins, perc_wins))
    print('mean all: {:.3f} ({:.3f}) mean win: {:.3f} ({:.3f})'
          .format(mean_all, std_dev_all, mean_win, std_dev_win))

    stats['all means'][fkey] += mean_all
    stats['all std dev'][fkey] += std_dev_all
    stats['wins means'][fkey] += mean_win
    stats['wins means std dev'][fkey] += std_dev_win
    stats['wins count'][fkey] += num_wins
    stats['wins percs'][fkey] += perc_wins

print()
NUM_SEEDS = len(SEEDS)
for stat in stats:
  for fkey in stat_dict:
    stats[stat][fkey] /= NUM_SEEDS
    print('mean {} {}: {:.3f}'.format(stat, fkey, stats[stat][fkey]))
