import numpy as np
import re
from configparser import ConfigParser


class ConfigReader(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.config = ConfigParser()
            self.config.read_file(f)


    def get_data(self):
        # Need to change here so that parameters (such as noise) are used
        factors = {}
        for k,v in self.config.items("Factors"):
            factors[k] = create_factor("../resources/{}.pgm".format(v))

        # Calculate PDF from formula
        values = {k.upper():v.values.astype(float) for k,v in factors.items()}
        formula = self.config.get("Density", "FORMULA")
        pdf = eval(formula, values)
        pdf = pdf / pdf.sum()

        return factors, pdf

    def get(self, section, key):
        return self.config.get(section, key)

class Factor(object):
    def __init__(self, values):
        height, width = values.shape
        self.width = width
        self.height = height

        # Inverting values so black pixels are highest
        # Currently making higher values matter more
        self.values = (255 - values.astype(float)) + 1
        self.values = self.values / self.values.max()
        # self.values[self.values < 150] = lesser / lesser.max()**2
        # self.values = values.astype(float)
        # self.values = (self.values - self.values.min()) / self.values.max()
        # self.values **= 2
        # self.values = self.values / self.values.sum()

        # distribution is normalized version of pixel values
        self.distribution = self.values / self.values.sum()

    def update_distribution(self):
        self.distribution = self.values / self.values.sum()



def create_factor(filename):
    values = read_pgm(filename)
    factor = Factor(values)
    return factor


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def write_pgm(filename, values):
    height, width = values.shape
    with open(filename, "w") as f:
        f.write("P2\n")
        f.write("{} {}\n".format(width, height))
        f.write("255\n")
        for v in values:
            for c in v:
                f.write(str(int(c)) + "\n")



if __name__ == '__main__':
    result = create_factor("/home/hcgs/ai/learning_testbed/resources/factor_shitty_circle.pgm")
