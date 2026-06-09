"""
Rapidly-exploring Random Tree Star (RRT*)
An asymptotically optimal sampling-based path planner.
"""

import math
import numpy as np
from typing import List, Tuple
from planning.environment import Node, KinematicEnvironment

class RRTStarPlanner:
    """
    Executes Kinematic Path Optimization using RRT*.
    Navigates a configuration space using random sampling, 
    dynamic radius nearest-neighbor searches, and tree rewiring.
    """
    def __init__(
        self, 
        env: KinematicEnvironment,
        start: Tuple[float, float], 
        goal: Tuple[float, float], 
        max_nodes: int = 20000,
        gamma: float = None,
        min_radius: float = 0.2
    ):
        self.env = env
        self.start = Node(*start)
        self.start.cost = 0.0
        
        self.goal = Node(*goal)
        self.max_nodes = max_nodes
        self.min_radius = min_radius
        self.gamma = gamma if gamma else 2 * math.sqrt(env.width * env.height)
        
        # Internal Tree Structures
        self.nodes: List[Node] = [self.start]
        self.children = {id(self.start): []}
        self.solved = False

    def get_dist(self, n1: Node, n2: Node) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def generate_random_node(self) -> Node:
        """Samples the configuration space. Biases toward the goal 5% of the time."""
        if np.random.rand() < 0.05:
            return Node(self.goal.x, self.goal.y)

        while True:
            x = np.random.uniform(-self.env.width / 2.0, self.env.width / 2.0)
            y = np.random.uniform(-self.env.height / 2.0, self.env.height / 2.0)
            rand_node = Node(x, y)

            if not self.env.is_inside_obstacles(rand_node):
                return rand_node

    def step_from_to(self, n0: Node, n1: Node, step_limit: float = 75.0) -> Node:
        """Generates a bounded kinematic step from n0 toward n1."""
        dist = self.get_dist(n0, n1)
        if dist <= step_limit:
            return Node(n1.x, n1.y)
        
        new_x = n0.x + ((n1.x - n0.x) / dist) * step_limit
        new_y = n0.y + ((n1.y - n0.y) / dist) * step_limit
        return Node(new_x, new_y)

    def propagate_cost_to_children(self, parent_node: Node):
        """Iteratively updates costs of all descendants after tree rewiring."""
        queue = [parent_node]
        while queue:
            current = queue.pop()
            for child in self.children.get(id(current), []):
                child.cost = current.cost + self.get_dist(current, child)
                queue.append(child)

    def rewire(self, near_nodes: List[Node], new_node: Node):
        """Optimizes the tree by routing nearby nodes through the newly added node."""
        for near_node in near_nodes:
            if near_node is new_node.parent or new_node.cost is None or near_node.parent is None:
                continue

            potential_cost = new_node.cost + self.get_dist(new_node, near_node)

            if (potential_cost < near_node.cost) and not self.env.is_collision_with_obstacles(new_node, near_node):
                self.children[id(near_node.parent)].remove(near_node)
                near_node.parent = new_node
                near_node.cost = potential_cost
                self.children[id(new_node)].append(near_node)
                self.propagate_cost_to_children(near_node)

    def compute_smooth_path(self, path: List[Node], iterations: int) -> List[Node]:
        """Recursively smooths a zig-zag RRT* path by attempting to bypass intermediate nodes."""
        if iterations <= 0 or len(path) <= 2:
            return path
            
        idx1 = np.random.randint(0, len(path) - 1)
        idx2 = np.random.randint(idx1 + 1, len(path))
        
        if idx2 - idx1 <= 1:
            return self.compute_smooth_path(path, iterations - 1)
            
        node_start, node_end = path[idx1], path[idx2]
        
        if not self.env.is_collision_with_obstacles(node_start, node_end):
            new_path = path[:idx1 + 1] + path[idx2:]
            return self.compute_smooth_path(new_path, iterations - 1)
            
        return self.compute_smooth_path(path, iterations - 1)

    def extract_path(self) -> List[Node]:
        """Backtracks from the goal to the start to formulate the final path."""
        path = []
        cur = self.goal
        while cur.parent is not None:
            path.append(cur)
            cur = cur.parent
        path.append(self.start)
        return path[::-1]

    def plan(self, min_nodes: int = 800) -> List[Tuple[float, float]]:
        """Executes the RRT* search loop. Returns a smoothed coordinate path."""
        nodes_added = 0

        while len(self.nodes) < self.max_nodes:
            rand_node = self.generate_random_node()
            nearest_node = min(self.nodes, key=lambda n: self.get_dist(n, rand_node))
            new_node = self.step_from_to(nearest_node, rand_node)
            
            if self.env.is_collision_with_obstacles(nearest_node, new_node):
                continue
                
            radius = max(self.min_radius, self.gamma * math.sqrt(math.log(len(self.nodes)) / len(self.nodes)))
            near_nodes = [n for n in self.nodes if self.get_dist(n, new_node) <= radius]
            
            best_parent = nearest_node
            min_cost = nearest_node.cost + self.get_dist(nearest_node, new_node)
            
            for near_node in near_nodes:
                if not self.env.is_collision_with_obstacles(near_node, new_node):
                    potential_cost = near_node.cost + self.get_dist(near_node, new_node)
                    if potential_cost < min_cost:
                        min_cost = potential_cost
                        best_parent = near_node
            
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            self.children[id(new_node)] = []
            self.children[id(best_parent)].append(new_node)
            nodes_added += 1
            
            self.rewire(near_nodes, new_node)

            if not self.solved and self.get_dist(new_node, self.goal) < 15 and not self.env.is_collision_with_obstacles(new_node, self.goal):
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + self.get_dist(new_node, self.goal)
                self.nodes.append(self.goal)
                self.children[id(self.goal)] = []
                self.children[id(new_node)].append(self.goal)
                self.solved = True
                print(f"Goal reached at {nodes_added} nodes. Optimizing path...")

            if self.solved and nodes_added >= min_nodes:
                break

        if self.solved:
            raw_path = self.extract_path()
            smoothed_nodes = self.compute_smooth_path(raw_path, 20)
            return [(n.x, n.y) for n in smoothed_nodes]
        else:
            print("No valid path found within node limits.")
            return []