"""Q-learning Agent.
WARNING: Values hardcoded for game with max 5 colors, ranks max 5."""
import random
import operator

from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.agents.qstate import QState
from hanabi_learning_environment.agents.q_util import action_to_hash, hash_to_action

EPS = 0.1

class QAgent(Agent):
  """Agent that applies Q-learning."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
    self.alpha = config.get('alpha', 0.1)
    self.gamma = config.get('gamma', 0.9)
    self.Q = {} # Values for new state-action pairs added lazily
    self.R = 0

  @staticmethod
  def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  def act(self, observation, get_next_state):
    """Act based on an observation. get_next_state() is used only for simulation."""
    # Checks if this game observer is the current player to make a move
    if observation['current_player_offset'] != 0:
      return None, -1
    S = QState(observation)
    # State not visited previously, initialise for all possible actions.
    if S not in self.Q:
      self.Q[S] = {action_to_hash(a): 0 for a in observation['legal_moves']}

    # Choose A from S using policy derived from Q -- here eps-greedy
    if random.random() <= EPS: # Random choice
      A_hash = random.choice(list(self.Q[S].keys()))
      print('random choice')
    else: # Greedy choice
      # Key corresponding to action with max value in Q[S] dict.
      A_hash = max(self.Q[S].items(), key=operator.itemgetter(1))[0]
    A = hash_to_action(A_hash)

    # Take (simulate) action A, observe R, S'
    S_new, R = get_next_state(A)
    max_Q_new = max(self.Q[S_new].values()) if S_new in self.Q else 0
    # Q(S, A) = Q(S, A) + alpha[R + gamma max_a Q(S', a) -Q(S, A)]
    self.Q[S][A_hash] += self.alpha * (R + self.gamma * max_Q_new - self.Q[S][A_hash])

    # This is the actual action to be performed in the environment.
    return A, R
