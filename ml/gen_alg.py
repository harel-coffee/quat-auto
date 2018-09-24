#!/usr/bin/env python3
import sys
import os
import argparse
import sys
from multiprocessing import Pool
import multiprocessing
import copy

import numpy as np
import random

class Individuum:
    _genom = np.array([])
    _fitness = 0
    _age = 0

    def __init__(self, number_genes=42):
        self._genom = np.array([0.0] * number_genes)

    def init_genom(self):
        for i in range(len(self._genom)):
            self._genom[i] = random.random()
        return self  # return self, simplifies construction

    def get_genom(self):
        return self._genom

    def __str__(self):
        return f"""{self.__class__.__name__}: fit: {self._fitness} """ # @genom: {self._genom}"""

    def print(self):
        return str(self)

    def calc_fitness(self):
        return 0

    def aging(self):
        self._age += 1
        return self

    def dead(self):
        return self._age > 80

    def get_fitness(self):
        return self._fitness

    def sex(self, other):
        c = copy.deepcopy(self)
        c._fitness = 0
        c._age = 0
        g = c.get_genom()
        go = other.get_genom()
        split = int(random.random() * len(g))
        c._genom = np.concatenate([g[0:split], go[split:]])
        return c

    def mutate(self, mutation_rate):
        for x in range(0, max(1, int(np.ceil(mutation_rate * len(self._genom))))):
            pos = random.randint(0, len(self._genom) - 1)
            self._genom[pos] = np.clip(self._genom[pos] +  (2 * random.random() - 1), 0, 1)
        self._fitness = 0


def _calc_fitness(individuum):
    individuum.calc_fitness()
    return individuum.get_fitness()


def calc_fitness(population, pool):
    """
    for i in population:
        i.calc_fitness()
    return
    """
    # calculate fitness, parallel
    r = pool.map(_calc_fitness, population)
    for i, x in enumerate(r):
        population[i]._fitness = x
    """
    """

def print_population(population):
    print("\n".join(map(str, population)))

def print_genes(population):
    print("\n".join(map(lambda x: str(x.get_genom()), population)))


def breed(parents, mutation_rate=0.5):
    childs = []
    while len(childs) < 2 * len(parents):
        # grab random parents and "merge them" to get a child
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        child = p1.sex(p2)
        child.mutate(mutation_rate)
        childs.append(child)

    return childs


def sort_by_fitness(population):
    return sorted(population, key=lambda x: x.get_fitness())


class HelloWorld(Individuum):
    target = np.array([ord(x) - 65 for x in "HelloWorld"], dtype=np.int)

    def __init__(self, number_genes=len("HelloWorld")):
        super().__init__(number_genes)

    def calc_fitness(self):
        nums = [int((122 - 65) * x) for x in self._genom]
        diff = (np.array(nums, dtype=np.int) - self.target) ** 2
        fit = diff.sum()
        self._fitness = fit
        return fit

    def print(self):
        return f"""
            fitness: {self._fitness},
            chars: {self.get_genom_as_char()},
            target: {self.target}
        """

    def get_genom_as_char(self):
        nums = [int((122 - 65) * x) for x in self._genom]
        chars = [str(chr(x + 65)) for x in nums]
        return "".join(chars)


import matplotlib.pyplot as plt

def main(_):
    random.seed(42)
    # initialization of initial population
    population_size = 200
    max_num_generations = 1000
    mutation_rate = 0.1

    INDIVIDUUM_CLASS = HelloWorld
    cpu_count = 4
    pool = Pool(processes=cpu_count)

    population = []
    for i in range(population_size):
        population.append(INDIVIDUUM_CLASS().init_genom())
    population = np.array(population)

    calc_fitness(population, pool)

    #print_genes(population)
    print_population(population)


    plt.show()

    axes = plt.gca()
    xdata = []
    ydata = []
    axes.set_xlim(0, max_num_generations)
    max_fitness = max([x.get_fitness() for x in population])
    axes.set_ylim(0, int(max_fitness * 0.1))
    line, = axes.plot(xdata, ydata, '-')

    num_generations = 0
    while num_generations < max_num_generations:
        print("generation:", num_generations, "last fittest:", population[0].get_fitness(), "genom:", population[0].get_genom_as_char())
        print([x.get_genom_as_char() for x in population[1:5]])
        xdata.append(num_generations)
        ydata.append(population[0].get_fitness())
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)

        # sort by fitness
        population = sort_by_fitness(population)

        # Select the best-fit individuals for reproduction. (Parents)
        # we take 25 % of all best individuals,
        parents = population[0:population_size // 4]
        # + random.sample(population[population_size // 2:], population_size // 4)
        # 25% from remaining random sampled

        # Breed new individuals through crossover and mutation operations to give birth to offspring.
        childs = breed(parents, mutation_rate)

        # Evaluate the individual fitness of new individuals.
        calc_fitness(childs, pool)

        population = population + childs
        population = sort_by_fitness(population)

        #population = population[0:10] + random.sample(population, population_size - 10)
        population = population[0:population_size]
        """

        # Replace least-fit population with new individuals.

        new_population = np.array(population[0:population_size])
        """
        num_generations += 1

    print(population[0].print())
    print(population[0].get_genom())

    return
    # argument parsing
    parser = argparse.ArgumentParser(description='train hyfu-- a no reference fume variant',
                                     epilog="stg7 2018",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("database", type=str, help="training database csv file (consists of video segment and rating value) or restricted yaml file")
    parser.add_argument("--feature_folder", type=str, default="features", help="folder for storing the features")
    parser.add_argument("--skip_feature_extraction", action="store_true", help="skip feature extraction step")
    parser.add_argument("--feaure_backtrack", action="store_true", help="backtrack all feature sets")
    parser.add_argument("--train_repetitions", type=int, default=1, help="number of repeatitions for training")
    parser.add_argument("--use_features_subset", action="store_true", help="use only a defined subset of features ({})".format(features_subset()))
    parser.add_argument("--model", type=str, default="models/hyfu.npz", help="output model")
    parser.add_argument("--mode", choices=[0,1], type=int, default=0, help="mode of model")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count() // 2, help='thread/cpu count')
    parser.add_argument("--validation_database", type=str, help="database that is used for validation")
    parser.add_argument("--feature_folder_validation", type=str, default="features", help="folder where validation features are stored")

    a = vars(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))