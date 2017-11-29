import random
from collections import Counter

class Shell(object):
    """
    Shells contain particles (ParamDist objects). The value is a shell is
    equal to its weight times the number of particles in it.

    The first step of pulling from a distribution involves a weighted pick
    based on the value of all the shells.
    The second step is a uniform pick from the particles in that shell.
    """
    def __init__(self, weight, chance):
        self.nodes = []
        self.node_indices = {}
        self.weight = weight
        self.chance = chance
        self.cumulative = 0
        self.above = None
        self.below = None

    def value(self, opposite=False):
        if opposite:
            return len(self.nodes) * 1/self.weight
        return len(self.nodes) * self.weight

    def remove_node(self, node):
        last_node = self.nodes[len(self.nodes) - 1]
        node_index = self.node_indices[node]
        self.node_indices.pop(node)
        self.nodes[node_index] = last_node
        if last_node != node:
            self.node_indices[last_node] = node_index

        self.nodes.pop()
        return node

    def add_node(self, node):
        self.nodes.append(node)
        self.node_indices[node] = len(self.nodes) - 1

    def update_node(self, node, is_reward):
        """
        Particle has a chance to move up or down a shell based on reward.
        """
        success = random.random() <= self.chance
        neighbor = [self.below, self.above][is_reward]

        # Success means the particle must move up or down a level
        if success and neighbor:
            node = self.remove_node(node)
            neighbor.add_node(node)
            return neighbor
        return self

    def pull_point(self):
        """ Uniformly pull point from distribution. """
        index = random.randint(0, len(self.nodes) - 1)
        point = self.nodes[index]
        return point


class ParamDist(object):
    """
    Each ParamDist represents a particular parameter (such as x = 5).
    It is also a distribution over other ParamDists conditioned on that
    particular parameter.

    For example, we could have that x = 5, and that ParamDist is a distribution
    over y given that x = 5.
    """
    def __init__(self, param_name, param_value, points=None, radius=0.0,
                 percolation=0.5, shell_size=20, is_exponential=False):
        if points is None:
            shell_size = 0
        self.radius = radius
        self.shells = []
        self.point_indices = {}
        self.next = None
        self.prev = None
        self.is_exponential = is_exponential
        self.shell_size = shell_size
        self.percolation = percolation
        self.construct_shells(shell_size)
        self.param_name = param_name
        self.param_value = param_value
        self.visited = 0
        self.visited_by = Counter()
        self.spread_by = Counter()
        self.spread = 0
        self.rewarded_by = Counter()

        # This ParamDist contains a distribution over other ParamDists.
        if points and shell_size:
            center_shell = self.shells[shell_size]
            for idx, point in enumerate(points):
                center_shell.nodes.append(point)
                center_shell.node_indices[point] = idx

            # Keep track of where all the ParamDists are located in the shells.
            # Initially, all points start in the middle.
            self.point_indices = {i:center_shell for i in points}


    def pull_from_dist(self, is_exploration=False):
        """
        Randomly pick a particle by doing the following:
            1: Pick a shell based on weight of shell and size
            2: Randomly pick a particle from within the shell

        If it's the exploration step, we favor picking shells that are at the
        center of the weighted distribution.
        """
        total = 0
        for shell in self.shells:
            if not shell.nodes:
                continue

            v = shell.value()
            if is_exploration and v > 1:
                v = 1/v
            total += v
            shell.cumulative = total

        pick = random.uniform(0, total)

        for shell in self.shells:
            if not shell.nodes:
                continue
            if pick <= shell.cumulative:
                point = shell.pull_point()
                return point.percolate()


    def do_rollout(self, rollout, is_exploration=False, record=True):
        """
        Recursively picks a point (parameter).
        If point contains a distribution over another parameter, repeat step.

        Args:
            rollout: list of chosen points (parameters)
            is_exploration: Change distribution if this is exploration step.
            record: Whether or not to record particle visitation frequency.
        """
        point = self.pull_from_dist(is_exploration)
        if record:
            point.visited += 1
            point.visited_by[self] += 1
        result = [self, point]
        rollout.append(result)
        if result[1].shell_size:
            result[1].do_rollout(rollout, is_exploration)


    def percolate(self):
        """ Randomly picks self or a neighbor based on percolation rate """
        if random.random() <= self.percolation:
            if random.random() >= 0.5:
                dest = self.prev or self.next or self
            else:
                dest = self.next or self.prev or self

            return dest.percolate()
        return self


    def construct_shells(self, steps):
        """
        Creates shells containing particles. Each shell has weight that is
        multiplied by the number of particles contained in it to get a value.
        """
        value_step = [( 2 * (x + 1)) for x in range(steps)]
        if self.is_exponential:
            value_step = [(2 ** (x + 1)) for x in range(steps)]
        for idx, i in enumerate(value_step[::-1]):
            weight = 1/i
            chance = 1/i
            self.shells.append(Shell(weight, chance))

        # Center layer where particles will start out
        self.shells.append(Shell(1, 0.5))

        for idx,i in enumerate(value_step):
            weight = i
            chance = 1/i
            self.shells.append(Shell(weight, chance))

        # Link shells together (from bottom to top)
        prev = None
        for shell in self.shells:
            if prev:
                prev.above = shell
            shell.below = prev
            prev = shell

    def print_shells(self):
        for idx, i in enumerate(self.shells):
            print("{}: {}".format(idx, len(i.nodes)))


def connect_neighbors(points):
    """ Connects points (parameters) together (used in percolation). """
    prev = None
    for point in points:
        if prev:
            prev.next = point
        point.prev = prev
        prev = point


def get_parameters(rollout):
    """ Extracts chosen parameters from rollout and returns dict with them. """
    params = {}
    for _, point in rollout:
        params[point.param_name] = point.param_value + random.uniform(-point.radius, point.radius)
    return params


def update_distributions(rollout, is_reward):
    """ Rewards or penalizes each distribution based on parameter success. """
    for parent, point in rollout:
        shell = parent.point_indices[point]
        parent.point_indices[point] = shell.update_node(point, is_reward)
