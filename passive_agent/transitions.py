## Count transitions from state s to state s' via given action
class TransitionMap:
    def __init__(self):
        self.transition_map = {}

    def get_dictionary(self):
        return self.transition_map
    
    def increment_transition(self, key):
        if key in self.transition_map:
            self.transition_map[key] += 1
        else:
            self.transition_map[key] = 1

    def get_transition_val(self, key):
        if key in self.transition_map:
            return self.transition_map[key]
        else:
            return 0

    def set_transition_prob(self, key, val):
        self.transition_map[key] = val