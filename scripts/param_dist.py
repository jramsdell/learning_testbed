import random
from collections import Counter

class Shell(object):
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
        # node = self.nodes[node_index]
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
        # success = random.random() <= 1 / self.weight
        success = random.random() <= self.chance
        neighbor = [self.below, self.above][is_reward]

        if success and neighbor:
            node = self.remove_node(node)
            neighbor.add_node(node)
            return neighbor
        return self

    def pull_point(self, radius=0):
        index = random.randint(0, len(self.nodes) - 1)
        # point = self.nodes[index] + random.uniform(-radius, radius)
        point = self.nodes[index]
        return point
        # return self.nodes[index], index

class ParamDist(object):
    def __init__(self, param_name, param_value, points=None, radius=0.0,
                 percolation=0.5, shell_size=20):
        if points is None:
            shell_size = 0
        self.radius = radius
        self.shells = []
        self.point_indices = {}
        self.next = None
        self.prev = None
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
        if points and shell_size:
            center_shell = self.shells[shell_size]
            for idx, point in enumerate(points):
                center_shell.nodes.append(point)
                center_shell.node_indices[point] = idx

            self.point_indices = {i:center_shell for i in points}


    def pull_from_dist(self, is_inverse=False):
        total = 0
        for shell in self.shells:
            if not shell.nodes:
                continue
            if is_inverse:
                v = shell.value()
                if v > 1:
                    v = 1/v
                # total += 1 / shell.value()
                total += v
            else:
                total += shell.value()
            shell.cumulative = total

        pick = random.uniform(0, total)

        for shell in self.shells:
            if not shell.nodes:
                continue
            if pick <= shell.cumulative:
                # point, index = shell.pull_point()
                point = shell.pull_point()
                return point.percolate()


    def do_rollout(self, rollout, is_inverse=False, record=True):
        point = self.pull_from_dist(is_inverse)
        if record:
            point.visited += 1
            point.visited_by[self] += 1
        result = [self, point]
        # history[result[0].param_name] = result
        rollout.append(result)
        if result[1].shell_size:
            result[1].do_rollout(rollout, is_inverse)


    def percolate(self):
        if random.random() <= self.percolation:
            if random.random() >= 0.5:
                dest = self.prev or self.next or self
            else:
                dest = self.next or self.prev or self

            return dest.percolate()
        return self


    def construct_shells(self, steps):
        # energy_step = [(1.25 ** (x + 1)) for x in range(steps)]
        energy_step = [(2 * (x + 1)) for x in range(steps)]
        for idx, i in enumerate(energy_step[::-1]):
            weight = 1/i
            chance = 1/i
            self.shells.append(Shell(weight, chance))

        self.shells.append(Shell(1, 0.5))

        for idx,i in enumerate(energy_step):
            weight = i
            chance = 1/i
            self.shells.append(Shell(weight, chance))

        prev = None
        for shell in self.shells:
            if prev:
                prev.above = shell
            shell.below = prev
            prev = shell

    def print_shells(self):
        for idx, i in enumerate(self.shells):
            print("{}: {}".format(idx, len(i.nodes)))


    def get_inverse_freq(self):
        freq_map = {}
        for k,v in self.visited_by.items():
            freq_map[k.param_value] = v / self.visited
        return freq_map



def connect_neighbors(points):
    prev = None
    for point in points:
        if prev:
            prev.next = point
        point.prev = prev
        prev = point


def get_parameters(rollout):
    params = {}
    for _, point in rollout:
        params[point.param_name] = point.param_value + random.uniform(-point.radius, point.radius)

    return params

def update_distributions(rollout, is_reward):
    for parent, point in rollout:
        shell = parent.point_indices[point]
        parent.point_indices[point] = shell.update_node(point, is_reward)

