"""Node used by the tree-based search algorithms (unchanged from Assignment 2A)."""


class Node:
    """One node in the search tree."""

    def __init__(self, node_id, x, y, parent=None, path_cost=0):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.parent = parent
        self.path_cost = path_cost

    def path(self):
        """Return this node's route from the origin as node IDs."""
        result = []
        current = self
        while current is not None:
            result.append(current.node_id)
            current = current.parent
        result.reverse()
        return result
