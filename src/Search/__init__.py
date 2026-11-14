"""
Task 4 - the tree-based search algorithms from Assignment 2A, ported to the TBRGS.

The six methods (DFS, BFS, GBFS, AS, CUS1, CUS2) are the Part A code with only three changes:
    * package imports (from .node import Node) instead of top-level imports
    * node IDs may be strings ('0970'), not only ints
    * straight-line distance and the heuristic come from Problem, so the same code works
      on the Part A test maps (x, y) and on the Boroondara map (lat, lon, seconds)

top_k.py adds Yen's algorithm on top of any of them to return up to k different routes.
"""
from .astar import astar
from .bfs import bfs
from .dfs import dfs
from .gbfs import greedy_best_first
from .ida_star import ida_star
from .iddfs import iddfs

# method code -> (search function, name shown to the user)
SEARCH_METHODS = {
    "AS": (astar, "A* search"),
    "BFS": (bfs, "Breadth-first search"),
    "DFS": (dfs, "Depth-first search"),
    "GBFS": (greedy_best_first, "Greedy best-first search"),
    "CUS1": (iddfs, "Iterative deepening DFS (CUS1)"),
    "CUS2": (ida_star, "Iterative deepening A* (CUS2)"),
}
