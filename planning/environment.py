"""
2D Kinematic Environment representation for path planning.
Handles configuration space boundaries and obstacle collision detection.
"""

from typing import List, Tuple, Optional

class Node:
    """Represents a discrete state in the configuration space."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['Node'] = None
        self.cost: float = float('inf')

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class KinematicEnvironment:
    """
    Manages the 2D configuration space, boundaries, and static obstacles.
    """
    def __init__(self, width: float, height: float, obstacles: List[List[Tuple[float, float]]]):
        self.width = width
        self.height = height
        
        # Convert coordinate tuples into Node objects for internal math
        self.obstacles = [[Node(*coord) for coord in obs] for obs in obstacles]

    def is_inside_obstacles(self, node: Node) -> bool:
        """Checks if a point lies strictly within any rectangular obstacle bounds."""
        for obstacle in self.obstacles:
            x1, y1 = obstacle[0].x, obstacle[0].y
            x2, y2 = obstacle[2].x, obstacle[2].y
            
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            if min_x <= node.x <= max_x and min_y <= node.y <= max_y:
                return True
        return False

    def is_intersect(self, p1: Node, p2: Node, p3: Node, p4: Node) -> bool:
        """Determines if line segment p1-p2 intersects p3-p4."""
        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def is_collision_with_obstacles(self, p1: Node, p2: Node) -> bool:
        """Checks if a kinematic step intersects with any obstacle edges."""
        for obstacle in self.obstacles:
            num_sides = len(obstacle)
            for idx in range(num_sides):
                side_start = obstacle[idx]
                side_end = obstacle[(idx + 1) % num_sides]
                if self.is_intersect(p1, p2, side_start, side_end):
                    return True
        return False