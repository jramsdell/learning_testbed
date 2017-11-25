import numpy as np
from param_dist import ParamDist
import random
from data_readers import ConfigReader, write_pgm
from sklearn.neighbors import KDTree

class PointSampler(object):
    def __init__(self, pdf):
        self.pdf = pdf
        self.tree = None
        self.pdf_flat = pdf.flatten()
        self.height, self.width = pdf.shape

    def sample(self, ntimes):
        """
        Samples ntimes coordinates from PDF and returns as list of tuples.
        """
        indices = np.asarray(range(self.height * self.width))
        choices = np.random.choice(indices, ntimes, p=self.pdf_flat)
        coords = list(map(self.coord_transform, choices))
        return coords

    def coord_transform(self, x):
        """
        Transforms 1D coordinates into a tuple representing 2D coordinates.
        """
        return (x % self.width, int(x / self.width))

    def create_sample_data(self, ntimes):
        """
        Creates a KD Tree (used for checking distance to points) from samples.
        """
        samples = self.sample(ntimes)
        self.tree = KDTree(samples)

    def query(self, point, radius):
        """
        Args:
            point: Tuple of x,y coordinates
            radius: Specifies radius around point to check for neighbors
        Returns:
            Number of points near sampled point
        """
        return self.tree.query_radius([point], radius, count_only=True)

    def write_pdf(self, filename, nsamples=100000):
        """
        Used for debugging pdf. Writes samples to a pgm.
        """
        samples = self.sample(nsamples)
        values = np.zeros((self.height, self.width), dtype=np.uint8)
        for sample in samples:
            values[sample[1]][sample[0]] = min(250, values[sample[1]][sample[0]] + 25)

        with open(filename, "w") as f:
            f.write("P2\n{} {}\n255\n".format(self.width, self.height))
            for i in np.nditer(values):
                f.write("{}\n".format(i))



class StatisticalModel(object):
    def __init__(self, manifold):
        self.extract_model(manifold)

    # def extract_model(self, manifold):
    #     for



class StatisticalManifold(object):
    def __init__(self, indicator):
        self.indicator = indicator
        self.layers = []
        self.top_layer = None
        self.rollout = None


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

    def do_rollout(self, is_inverse=False):
        self.rollout = []
        self.top_layer.do_rollout(self.rollout, is_inverse)

    def get_parameters(self, rollout):
        params = {}
        for _, point in rollout:
            params[point.param_name] = \
                point.param_value + random.uniform(-point.radius, point.radius)
        return params

    def do_backup(self, is_inverse=False):
        params = self.get_parameters(self.rollout)
        is_reward = self.indicator(params)
        if is_inverse:
            is_reward = not is_reward

        for parent, point in self.rollout:
            shell = parent.point_indices[point]
            parent.point_indices[point] = shell.update_node(point, is_reward)

    def train(self, ntimes):
        for i in range(ntimes):
            is_inverse = i % 2 == 0
            self.do_rollout(is_inverse=is_inverse)
            self.do_backup(is_inverse=is_inverse)

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
            self.do_rollout()
            params = self.get_parameters(self.rollout)
            x,y = int(params["X"]), int(params["Y"])
            try:
                values[y][x] += 1
            except IndexError:
                continue

        values = values / values.sum()
        return values



class ExperimentRunner(object):
    def __init__(self, config_path):
        self.pdf_manifold = None

        self.config = ConfigReader(config_path)
        self.factors, self.pdf = self.config.get_data()
        seed = int(self.config.get("Experiment", "SEED"))
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
        samples = int(self.config.get("Density", "SAMPLES"))
        self.sampler = PointSampler(self.pdf)
        self.sampler.create_sample_data(samples)
        self.initialize_pdf_manifold()

    def run(self):
        train_runs = int(self.config.get("Experiment", "TRAIN_RUNS"))
        self.pdf_manifold.train(train_runs)
        # self.pdf_manifold.print_results(100)
        height, width = self.sampler.height, self.sampler.width
        values = self.pdf_manifold.estimate_frequencies(
            50000, height, width)

        values = values / values.max()
        values = (values * 255).astype(np.uint8)
        write_pgm("manifold_estimate.pgm", values)

    def initialize_pdf_manifold(self):
        height, width = self.sampler.height, self.sampler.width
        interval = float(self.config.get("Experiment", "INTERVAL"))
        point_radius = float(self.config.get("Experiment", "POINT_RADIUS"))
        percolation = float(self.config.get("Experiment", "PERCOLATION"))
        indicator = self.get_indicator()

        self.pdf_manifold = StatisticalManifold(indicator)

        xs = self.get_interval_values(interval, width)
        ys = self.get_interval_values(interval, height)

        self.pdf_manifold.add_layer(
            "Y", ys, radius=point_radius, percolation=percolation)

        self.pdf_manifold.add_layer(
            "X", xs, radius=point_radius, percolation=percolation)

        self.pdf_manifold.add_top_layer()


    def get_interval_values(self, interval, length):
        values = []
        istep = length / interval
        for step in range(int(interval)):
            values.append(float(step * istep))
        return values

    def get_indicator(self):
        reward_radius = float(self.config.get("Experiment", "REWARD_RADIUS"))
        sampler = self.sampler

        def indicator(params):
            point = (params["X"], params["Y"])
            return sampler.query(point, reward_radius)[0] > 0

        return indicator






if __name__ == '__main__':
    m = StatisticalManifold(lambda x: (10 <= x["Y"] + x["X"] <= 12) )
    m.add_layer("Y", range(1, 100), percolation=0.0)
    m.add_layer("X", range(1, 100), percolation=0.0)
    m.add_top_layer()
    m.train(10000)
    # m.print_results(20)
    # errors = m.evaluate_error(1000)
    # print(errors)
    # print(m.get_parameters(m.rollout))

    e = ExperimentRunner("/home/hcgs/ai/learning_testbed/configurations/config.cfg")
    e.run()




