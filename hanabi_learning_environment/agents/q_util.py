"""Helper standalone functions for Q-learning agent."""

from hanabi_learning_environment.agents.qstate import QState
from hanabi_learning_environment.rl_env import HanabiEnv
from hanabi_learning_environment.pyhanabi import HanabiMoveType

# NOTE: Different colormap values than in qstate.py!
COLORMAP = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
REV_COLORMAP = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
NUM_COLORS = 5
NUM_CARDS = 5 # Max number of cards held by a player
NUM_PLAYERS = 5
NUM_RANKS = 5
FINAL_SCORE_MULTIPLIER = 1000
CORRECT_PLAYED_CARD_REWARD = 5
DISCARD_LAST_CARD_REWARD = -3
DISCARD_PLAYED_CARD_REWARD = 5
DISCARD_NOT_PLAYED_IN_GAME_CARD_REWARD = 3
BASE_REVEL_REWARD = 3
GAME_LOST_REWARD = -25000

def get_next_state(state, player_id, action, hanabi_game):
  """Simulate the action, get new state and reward"""
  action = HanabiEnv.build_move_static(action)
  prev_state = state.copy()
  state.apply_move(action)
  observation = HanabiEnv.extract_dict_static(player_id, state.observation(player_id), state)
  reward = calculate_reward(prev_state, state, action, hanabi_game)
  return QState(observation), reward

def calculate_reward(prev_state, cur_state, action, hanabi_game):
  """Get reward for taking the action from prev_state to cur_state"""
  if cur_state.is_terminal():
    # If current state is terminal return much bigger reward than in any other case.
    # Reward based on final points.
    if cur_state.score() == 0:
      return GAME_LOST_REWARD
    return cur_state.score() * FINAL_SCORE_MULTIPLIER

  if action.type() == HanabiMoveType.PLAY:
    # If discard pile size is same in both states, card was played properly.
    if len(cur_state.discard_pile()) == len(prev_state.discard_pile()):
      return CORRECT_PLAYED_CARD_REWARD
    return -CORRECT_PLAYED_CARD_REWARD

  if action.type() == HanabiMoveType.DISCARD:
    cards_count = hanabi_game.num_cards(action.color(), action.rank())
    cards_on_discard_pile = 0
    for card in cur_state.discard_pile():
      if card.color() == action.color() and card.rank() == action.rank():
        cards_on_discard_pile += 1

    # If all same type cards in game are on discard pile, one fireworks pile cannot be completed.
    if cards_count == cards_on_discard_pile:
      return DISCARD_LAST_CARD_REWARD

    # If discarded card was already played on fireworks pile, discard move was perfect :)
    if cur_state.fireworks()[action.color()] - 1 >= action.rank():
      return DISCARD_PLAYED_CARD_REWARD

    return DISCARD_NOT_PLAYED_IN_GAME_CARD_REWARD

  if action.type() == HanabiMoveType.REVEAL_COLOR or action.type() == HanabiMoveType.REVEAL_RANK:
    return BASE_REVEL_REWARD

  raise Exception("Action type is illegal: " + str(action.type()))

def action_to_hash(action):
  """Action is a dict whose content is determined by the action type.
  This is a copy of HanabiGame::GetMoveUid()."""
  if action['action_type'] == 'DISCARD':
    return action['card_index']
  if action['action_type'] == 'PLAY':
    return NUM_CARDS + action['card_index']
  if action['action_type'] == 'REVEAL_COLOR':
    return NUM_CARDS + NUM_CARDS + (action['target_offset'] - 1) * NUM_COLORS + \
           COLORMAP[action['color']]
  if action['action_type'] == 'REVEAL_RANK':
    return NUM_CARDS + NUM_CARDS + (NUM_PLAYERS - 1) * NUM_COLORS + NUM_COLORS + \
           (action['target_offset'] - 1) * NUM_RANKS + action['rank']
  return -1

def hash_to_action(a_hash):
  """Deciphering the hash to a game action."""
  if a_hash == -1:
    raise ValueError('Invalid action hash')
  if a_hash < NUM_CARDS:
    return {'action_type': 'DISCARD', 'card_index': a_hash}
  if a_hash < NUM_CARDS + NUM_CARDS:
    return {'action_type': 'PLAY', 'card_index': a_hash - NUM_CARDS}
  if a_hash < NUM_CARDS + NUM_CARDS + (NUM_PLAYERS - 1) * NUM_COLORS + NUM_COLORS:
    target = (a_hash - NUM_CARDS - NUM_CARDS) // NUM_COLORS + 1
    color = a_hash - NUM_CARDS - NUM_CARDS - (target - 1) * NUM_COLORS
    return {'action_type': 'REVEAL_COLOR', 'target_offset': target,
            'color': REV_COLORMAP[color]}
  target = (a_hash - NUM_CARDS - NUM_CARDS - (NUM_PLAYERS - 1) * NUM_COLORS - NUM_COLORS) // \
      NUM_RANKS + 1
  rank = a_hash - NUM_CARDS - NUM_CARDS - (NUM_PLAYERS - 1) * NUM_COLORS - \
      NUM_COLORS - (target - 1) * NUM_RANKS
  return {'action_type': 'REVEAL_RANK', 'target_offset': target, 'rank': rank}
