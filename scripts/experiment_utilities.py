import numpy as np
import math
from param_dist import ParamDist
import random
from data_readers import ConfigReader, write_pgm
from sklearn.neighbors import KDTree
from statistical_manifold import StatisticalManifold

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
        self.write_pgm_samples(samples, "samples_used.pgm")
        self.tree = KDTree(samples)

    def write_sample_data(self, samples, filename):
        with open(filename, "w") as f:
            for sample in samples:
                f.write("{} {}\n".format(*sample))


    def load_sample_data(self, filename):
        samples = []
        with open(filename) as f:
            for line in f:
                x,y = line.split()
                samples.append((int(x), int(y)))
        return samples


    def write_pgm_samples(self, samples, filename):
        values = np.zeros((self.height, self.width), dtype=np.uint8)
        for sample in samples:
            values[sample[1]][sample[0]] = 255

        with open(filename, "w") as f:
            f.write("P2\n{} {}\n255\n".format(self.width, self.height))
            for i in np.nditer(values):
                f.write("{}\n".format(i))



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
        height, width = self.sampler.height, self.sampler.width
        self.run_pdf_manifold(height, width)
        self.run_observer_manifold(height, width)



    def run_pdf_manifold(self, height, width):
        train_runs = int(self.config.get("Experiment", "TRAIN_RUNS"))
        self.pdf_manifold.train(train_runs)

        values = self.pdf_manifold.estimate_frequencies(
            50000, height, width)

        values = values / values.max()
        self.sampler.write_pdf("test_pdf.pgm")
        # values *= 30
        values = (values * 255).astype(np.uint8)
        write_pgm("manifold_estimate.pgm", values)

        # d1 = self.factors["f5"].distribution
        # d2 = self.factors["f6"].distribution
        # dfinal = (d2 * 0.5 + d1 * 0.5)**2
        # dfinal /= dfinal.sum()
        # f1_acc = self.get_feature_accessor(dfinal)
        # self_exp = self.get_self_expectation(dfinal)
        # # print(math.log(abs(self.pdf_manifold.get_expectation(f1_acc) / self_exp)))
        # print((abs(self.pdf_manifold.get_expectation(f1_acc) * self_exp)))
        #
        # d1 = self.factors["f5"].distribution
        # d2 = self.factors["f6"].distribution
        # dfinal = (d2 * 0.65 + d1 * 0.35)**2
        # dfinal /= dfinal.sum()
        # f1_acc = self.get_feature_accessor(dfinal)
        # self_exp = self.get_self_expectation(dfinal)
        # print(abs(self.pdf_manifold.get_expectation(f1_acc) * self_exp))



    def run_observer_manifold(self, height, width):
            self.pdf_manifold.create_trajectory(2000)

            self.initialize_observer_manifold(self.factors["f5"].values,
                                          self.factors["f6"].values)
            self.observer_manifold.train(5000)

            # self.observer_manifold.create_trajectory(1000)
            # factor1 = self.factors["f5"].values
            # factor2 = self.factors["f6"].values
            # values = np.zeros(factor1.shape)
            # for params in self.observer_manifold.trajectory:
            #     mixture = factor1 * params["X"] + factor2 * params["Y"]
            #     mixture = mixture / max(0.001, mixture.sum())
            #     values += mixture
            #     accessor = self.get_feature_accessor(mixture)
            #     res = self.pdf_manifold.get_expectation(accessor)


            values = self.observer_manifold.estimate_frequencies(
                50000, height, width)

            values = values / values.max()
            values = (values * 255).astype(np.uint8)
            self.observer_manifold.print_center()
            self.observer_manifold.print_results(10)
            write_pgm("observer_manifold_estimate.pgm", values)





    def initialize_pdf_manifold(self):
        height, width = self.sampler.height, self.sampler.width
        interval = float(self.config.get("Experiment", "INTERVAL"))
        point_radius = float(self.config.get("Experiment", "POINT_RADIUS"))
        percolation = float(self.config.get("Experiment", "PERCOLATION"))
        spread = int(self.config.get("Experiment", "SPREAD"))
        indicator = self.get_indicator()
        # indicator = lambda p: (p["Y"] <= 120) or (p["X"] <= 160)
        # indicator = lambda p: True
        # indicator = lambda p: 180 <= math.sqrt(p["X"]**2 + p["Y"]**2) <= 200 or (150 < p["X"] < 300)

        self.pdf_manifold = StatisticalManifold(indicator, spread=spread)

        xs = self.get_interval_values(interval, width)
        ys = self.get_interval_values(interval, height)

        self.pdf_manifold.add_layer(
            "Y", ys, radius=point_radius, percolation=percolation)

        self.pdf_manifold.add_layer(
            "X", xs, radius=point_radius, percolation=percolation)
        self.pdf_manifold.add_top_layer()

    def initialize_observer_manifold(self, factor1, factor2):
        height, width = self.sampler.height, self.sampler.width
        interval = float(self.config.get("Experiment", "INTERVAL"))
        point_radius = float(self.config.get("Experiment", "POINT_RADIUS"))
        percolation = float(self.config.get("Experiment", "PERCOLATION"))
        spread = int(self.config.get("Experiment", "SPREAD"))
        highest = 0
        total = 0
        nsamples = 0

        def evaluate_mixture(params):
            nonlocal total, nsamples
            mixture = factor1 * params["X"] + (factor2 * params["Y"])
            mixture = mixture / mixture.sum()
            accessor = self.get_feature_accessor(mixture)
            expect = self.pdf_manifold.get_expectation(accessor)
            res = abs((expect - self.get_self_expectation(mixture))) / self.get_self_expectation(mixture)
            # res = expect
            if nsamples == 0:
                total = res
                nsamples += 1
                return True

            if random.random() <= ((total / nsamples) / res)**5:
                total += res
                nsamples += 1
                return True

            return False

        self.observer_manifold = StatisticalManifold(evaluate_mixture, spread=20)
        xs = self.get_interval_values(200, width)
        ys = self.get_interval_values(200, height)

        self.observer_manifold.add_layer(
            "Y", ys, radius=height / 300, percolation=percolation)

        self.observer_manifold.add_layer(
            "X", xs, radius=width / 300, percolation=percolation)
        self.observer_manifold.add_top_layer()


    def get_interval_values(self, interval, length):
        values = []
        istep = length / interval
        for step in range(int(interval)):
            values.append(float(step * istep))
        return values

    def get_indicator(self):
        reward_radius = float(self.config.get("Experiment", "REWARD_RADIUS"))
        sampler = self.sampler
        total = 0
        nsamples = 0

        def indicator(params):
            nonlocal total, nsamples
            point = (params["X"], params["Y"])
            neighbors = sampler.query(point, reward_radius)[0]
            if random.random() <= (neighbors / (max(total, 1) / max(nsamples, 1)))**1.5:
                nsamples += 1
                total += neighbors
                return True
            return False
            # highest = max(highest, neighbors)
            # return random.random() <= (neighbors / highest)**2

        return indicator

    def get_feature_accessor(self, feature):
        def accessor(params):
            return feature[params["Y"]][params["X"]]
        return accessor

    def get_self_expectation(self, feature):
        return (feature**2).sum()




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




