"""Hashable game state for Q-learning."""

COLORMAP = {None: 10, 'R': 20, 'Y': 30, 'G': 40, 'W': 50, 'B': 60}

def card_to_int(card):
  """Helper to hash a single card which is a dict of form: {'color': color, 'rank': value}"""
  rank = card['rank'] + 2 # due to -1 if card rank is not known.
  return COLORMAP[card['color']] + rank # card['rank']

class QState():
  """Represents state of the game from the Agent perspective i.e. observations.
  A hashable class to make state keys for Q dict."""

  def __init__(self, observation):
    self.hands = observation['observed_hands']
    self.discard_pile = observation['discard_pile']
    self.fireworks = observation['fireworks']

    # Hands are a list of lists of card dicts
    self.h_hands = tuple([tuple(sorted([card_to_int(card) for card in hand]))
                         for hand in observation['observed_hands']])

    # Note: the order of card on the pile assumend to be not important.
    # We lose the information which card was discarded most recently.
    # Discard pile is a list of card dicts.
    # We convert it to a hashable tuple of ints.
    # The sorting makes hashing indifferent to the order of cards in the pile.
    self.h_discard_pile = tuple(sorted([card_to_int(card) for card in observation['discard_pile']]))

    # Encoding dict of {'color': count} to an int
    self.h_fireworks = 0
    for f_ind, f in enumerate(observation['fireworks'].values()): # dict
      self.h_fireworks = self.h_fireworks + f * (f_ind + 1) * 10

    self.info_tokens = observation['information_tokens'] # int
    self.life_tokens = observation['life_tokens'] # int


  def __hash__(self):
    return hash((self.h_hands, self.h_discard_pile, self.h_fireworks,
            self.info_tokens, self.life_tokens))

  def __eq__(self, other):
    self_tuple = (self.h_hands, self.h_discard_pile, self.h_fireworks,
                    self.info_tokens, self.life_tokens)
    other_tuple = (other.h_hands, other.h_discard_pile, other.h_fireworks,
                    other.info_tokens, other.life_tokens)
    return self_tuple == other_tuple

  def __ne__(self, other):
    # Not strictly necessary, but to avoid having both x==y and x!=y
    # True at the same time
    return not self == other
