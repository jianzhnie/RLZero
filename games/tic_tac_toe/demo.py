import numpy as np
import pyspiel

[
    '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__',
    '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__',
    '__init__', '__init_subclass__', '__le__', '__lt__', '__module__',
    '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
    '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__',
    'action_to_string', 'deserialize_state', 'get_parameters', 'get_type',
    'information_state_tensor_layout', 'information_state_tensor_shape',
    'information_state_tensor_size', 'make_observer',
    'max_chance_nodes_in_history', 'max_chance_outcomes', 'max_game_length',
    'max_history_length', 'max_move_number', 'max_utility', 'min_utility',
    'new_initial_state', 'new_initial_state_for_population',
    'new_initial_states', 'num_distinct_actions', 'num_players',
    'observation_tensor_layout', 'observation_tensor_shape',
    'observation_tensor_size', 'policy_tensor_shape', 'utility_sum'
]


def main():
    game = pyspiel.load_game('kuhn_poker')
    state = game.new_initial_state()
    while not state.is_terminal():
        legal_actions = state.legal_actions()
        print('legal_actions:', legal_actions)
        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            print('outcomes_with_probs:', outcomes_with_probs)
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
            print('state:', state)
        else:
            # The algorithm can pick an action based on an observation (fully observable
            # games) or an information state (information available for that player)
            # We arbitrarily select the first available action as an example.
            action = legal_actions[0]
            state.apply_action(action)


if __name__ == '__main__':
    main()
