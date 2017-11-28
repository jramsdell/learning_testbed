import random
import numpy as np
from param_dist import ParamDist

class StatisticalModel(object):
    def __init__(self, manifold):
        self.extract_model(manifold)

        # def extract_model(self, manifold):
        #     for



class StatisticalManifold(object):
    def __init__(self, indicator, spread=0):
        self.indicator = indicator
        self.layers = []
        self.top_layer = None
        self.rollout = None
        self.spread = spread
        self.backups = 0
        self.rewards = 0
        self.trajectory = []


    def add_layer(self, param_name, param_values, radius=0.0,
                  percolation=0.5, shell_size=20):

        # If we already have a layer, we will link points to new layer
        points = None
        if self.layers:
            points = self.layers[-1]

        # create points for new layer and the link them
        layer = []
        for i in param_values:
            layer.append(
                ParamDist(param_name, i, points=points, radius=radius,
                          percolation=percolation, shell_size=shell_size))

        self.connect_neighbors(layer)
        self.layers.append(layer)

    def add_top_layer(self, shell_size=20):
        self.top_layer = \
            ParamDist("Top", None, self.layers[-1], shell_size=shell_size)


    def connect_neighbors(self, points):
        prev = None
        for point in points:
            if prev:
                prev.next = point
            point.prev = prev
            prev = point

    def do_rollout(self, is_inverse=False, record=True):
        self.rollout = []
        self.top_layer.do_rollout(self.rollout, is_inverse, record=record)

    def get_parameters(self, rollout):
        params = {}
        for _, point in rollout:
            params[point.param_name] = \
                point.param_value + random.uniform(-point.radius, point.radius)
        return params


    def choose_reward(self, visited_by):
        total = 0
        best = {}
        for k,v in visited_by.items():
            total += v
            best[k] = total

        pick = random.uniform(0, total)
        for k,v in visited_by.items():
            if pick <= best[k]:
                return k





    def do_backup(self, is_inverse=False):
        self.backups += 1
        params = self.get_parameters(self.rollout)
        is_reward = self.indicator(params)
        self.rewards += is_reward
        # if is_inverse:
        #     is_reward = not is_reward

        for parent, point in self.rollout:
            if is_reward:
                point.rewarded_by[parent] += 1

                for i in range(self.spread):
                    best = self.choose_reward(point.rewarded_by)
                    point.spread_by[best] += 1
                    shell = best.point_indices[point]
                    best.point_indices[point] = shell.update_node(point, is_reward)
                    # point.rewarded_by[best] += 1
                    # point.visited += 1


            shell = parent.point_indices[point]
            parent.point_indices[point] = shell.update_node(point, is_reward)
            # ugly... ignore for now
            if point.param_name == "Y":
                for p in [parent.next, parent.prev]:

                    if p:
                        shell = p.point_indices[point]
                        p.point_indices[point] = shell.update_node(point, is_reward)

                        for p2 in [p.next, p.prev]:
                            if p2:
                                shell = p2.point_indices[point]
                                p2.point_indices[point] = shell.update_node(point, is_reward)
                                for p3 in [p2.next, p2.prev]:
                                    if p3:
                                        shell = p3.point_indices[point]
                                        p3.point_indices[point] = shell.update_node(point, is_reward)

                                        for p4 in [p3.next, p3.prev]:
                                            if p4:
                                                shell = p4.point_indices[point]
                                                p4.point_indices[point] = shell.update_node(point, is_reward)



    def train(self, ntimes):
        for i in range(ntimes):
            is_inverse = i % 2 == 0
            # is_inverse = False
            self.do_rollout(is_inverse=is_inverse)
            self.do_backup(is_inverse=is_inverse)


    def print_center(self):
        total = None
        for i in range(10000):
            self.do_rollout(record=False)
            params = self.get_parameters(self.rollout)
            if total is None:
                total = params
            else:
                for k,v in params.items():
                    total[k] += v

        for k,v in total.items():
            total[k] = v / 10000
        print(total)

    def self_expectation(self):
        pass

    def create_trajectory(self, ntimes):
        self.trajectory = []
        for i in range(ntimes):
            self.do_rollout(record=False)
            params = self.get_parameters(self.rollout)
            if params["X"] >= 240:
                params["X"] = 239
            elif params["X"] < 0:
                params["X"] = 0
            if params["Y"] >= 240:
                params["Y"] = 239
            elif params["Y"] < 0:
                params["Y"] = 0
            params["Y"] = int(params["Y"])
            params["X"] = int(params["X"])
            self.trajectory.append(params)


    def get_expectation(self, function):
        total = 0
        for params in self.trajectory:
            total += function(params)
        return total / len(self.trajectory)


    def evaluate_error(self, ntimes):
        errors = 0
        for i in range(ntimes):
            self.do_rollout()
            errors += not self.indicator(self.get_parameters(self.rollout))
        return errors

    def print_results(self, ntimes):
        for i in range(ntimes):
            self.do_rollout()
            params = self.get_parameters(self.rollout)
            output = ""
            for k,v in sorted(params.items()):
                if k == "Top":
                    continue
                output += "{}: {}\t".format(k,v)
            print(output)


    def estimate_frequencies(self, ntimes, height, width):
        values = np.zeros((height, width))
        for i in range(ntimes):
            self.do_rollout(record=False)
            params = self.get_parameters(self.rollout)
            x,y = int(params["X"]), int(params["Y"])
            try:
                values[y][x] += 1
            except IndexError:
                continue

        # values = values / values.sum()
        return values




