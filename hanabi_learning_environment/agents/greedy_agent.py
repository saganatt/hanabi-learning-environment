"""Greedy Agent."""
import random
import operator

from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.agents.qstate import QState
from hanabi_learning_environment.agents.q_util import action_to_hash, hash_to_action


class GreedyAgent(Agent):
  """Greedy agent for given policy."""

  def __init__(self, config, Q, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    self.Q = Q

  def act(self, observation, get_next_state):
    """Act based on an observation. get_next_state() is used only for simulation."""
    # Checks if this game observer is the current player to make a move
    if observation['current_player_offset'] != 0:
      return None, -1

    S = QState(observation)
    choose_randomly = False
    actions = {action_to_hash(a): 0 for a in observation['legal_moves']}

    # State not visited previously, initialise for all possible actions.
    if S not in self.Q:
      print('Greedy agent encountering a new state')
      choose_randomly = True
      self.Q[S] = actions
    else:
      # In case of simple state settings (less accurate hashing), QAgent might be in
      # a bit different state and add some actions forbidden in the actual current
      # GreedyAgent state. So we pick only actions that are contained in the legal moves.
      new_actions = {a: self.Q[S][a] for a in self.Q[S].keys() & actions}
      if new_actions: # If not empty, choose from valid subset of Q[S]
        actions = new_actions
      else:
        choose_randomly = True # Force random choice as there is no valid action available in Q

    # Choose A from S using policy derived from Q
    # In case of less accurate hashing, choice is restricted to subset of valid actions
    if choose_randomly: # Random choice
      A_hash = random.choice(list(actions.keys()))
    else: # Greedy choice
      # Key corresponding to action with max value in Q[S] dict.
      A_hash = max(actions.items(), key=operator.itemgetter(1))[0]
    A = hash_to_action(A_hash)

    _, R = get_next_state(A)

    return A, R
