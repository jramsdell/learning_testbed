import random
from collections import Counter

class Shell(object):
    def __init__(self, weight, min_energy, max_energy):
        self.nodes = []
        self.node_indices = {}
        self.weight = weight
        self.min_energy = min_energy
        self.max_energy = max_energy
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

    # def update_node(self, node, shells):
    #     if node.energy < self.min_energy:
    #         node.level -= 1
    #         self.nodes.remove(node)
    #         shells[node.level].nodes.append(node)
    #     elif node.energy > self.max_energy:
    #         node.level += 1
    #         self.nodes.remove(node)
    #         shells[node.level].nodes.append(node)

    def update_node(self, node, is_reward):
        success = random.random() <= 1 / self.weight
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
                total += 1 / shell.value()
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


    def do_rollout(self, rollout, is_inverse=False):
        point = self.pull_from_dist(is_inverse)
        point.visited += 1
        point.visited_by[self] += 1
        result = [self, point]
        # history[result[0].param_name] = result
        rollout.append(result)
        if result[1].shell_size:
            result[1].do_rollout(rollout, is_inverse)


    def percolate(self):
        if random.random() <= self.percolation:
            dest = [self.prev, self.next][random.randint(0,1)]
            if dest:
                return dest.percolate()

        return self


    def construct_shells(self, steps):
        # energy_step = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        energy_step = [(2 ** (x + 1)) for x in range(steps)]
        for idx, i in enumerate(energy_step[::-1]):
            # weight = 1/(idx * 4 + 2)
            # weight = 1/i**3
            weight = 1/i
            min_energy = -i
            max_energy = -(int(i/2))
            self.shells.append(Shell(weight, min_energy, max_energy))

        self.shells.append(Shell(1, 0, 0))

        for idx,i in enumerate(energy_step):
            # weight = idx * 4 + 1
            weight = i + 1
            min_energy = int(i/2)
            max_energy = i
            self.shells.append(Shell(weight, min_energy, max_energy))

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

if __name__ == '__main__':
    ys = []
    for i in range(100):
        ys.append(ParamDist("Y", i + 1, None, 0.5, percolation=0.95))
    connect_neighbors(ys)

    xs = []
    for i in range(100):
        xs.append(ParamDist("X", i + 1, ys, percolation=0.95))
    connect_neighbors(xs)

    multi_dist = ParamDist("", 0, xs)
    # history = []
    indicator = lambda p: (10 <= (p["X"] + p["Y"]) <= 40) or ((65 <= (p["X"] + p["Y"]) <= 100))
    for i in range(1000):
        rollout = []
        multi_dist.do_rollout(rollout)
        params = get_parameters(rollout)
        success = indicator(params)
        update_distributions(rollout, success)

    errors = 0
    for i in range(100):
        rollout = []
        multi_dist.do_rollout(rollout)
        params = get_parameters(rollout)
        errors += not indicator(params)

        # for x in rollout:
        #     for i in x[1].visited_by.most_common(1):
        #         print("{}: {}".format(i[0].param_value, i[1] / x[1].visited))
            # invf = x[1].get_inverse_freq()
            # for k,v in sorted(invf.items(), key=lambda x: x[1]):
            #     print(k,v)



    # p = ParamDist([i + 1 for i in range(100)], "X", 1.0)
    # # p.child = ParamDist([(i + 1)**2 for i in range(10)], "Y", 0.5)
    # p.child = ParamDist([i + 1 for i in range(100)], "Y", 0.5)
    # indicator = lambda x,y: (30 <= x + y <= 45) or (y >= 80)

    # for i in range(100000):
    #     results = []
    #     p.do_rollout(results)
    #
    #     x = results[0][1]
    #     y = results[1][1]
    #     is_reward = indicator(x, y)
    #
    #     for result in results[::-1]:
    #         _, _, index, shell = result
    #         shell.update_node(index, is_reward)
    #
    # error = 0
    # for i in range(100):
    #     results = p.do_rollout([])
    #     x = results[0][1]
    #     y = results[1][1]
    #     comb = x + y
    #     print("{}, {}, {}".format(x, y, x + y))
    #     if not indicator(x,y):
    #         error += 1
    #
    # # p.print_shells()
    # # p.child.print_shells()
    # print("Error: {}".format(error))

