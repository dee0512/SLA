import numpy as np


class States:
    def __init__(self, removed_indices=[]):
        self.removed_indices = removed_indices

    def get_state(self, state):
        return np.delete(state, self.removed_indices)
