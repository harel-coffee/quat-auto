#!/usr/bin/env python3
import sys
import os
import argparse
import sys
from multiprocessing import Pool
import multiprocessing
import copy
import random

import numpy as np
import matplotlib.pyplot as plt


class Individual:
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



class HelloWorld(Individual):
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


def _calc_fitness(individual):
    individual.calc_fitness()
    return individual.get_fitness()




class genetic_evolution:
    def __init__(self, population_size, mutation_rate, max_num_generations, INDIVIDUAL_CLASS, cpu_count=multiprocessing.cpu_count(), verbose=False, live_plot=True):
         # create multiprocessing pool
        self._pool = Pool(processes=cpu_count)
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._max_num_generations = max_num_generations
        self._INDIVIDUAL_CLASS = INDIVIDUAL_CLASS
        self._cpu_count = cpu_count

        # create initial population
        self._population = []
        for i in range(self._population_size):
            self._population.append(self._INDIVIDUAL_CLASS().init_genom())
        self._population = np.array(self._population)

        self._calc_fitness_of_all(self._population)
        if verbose:
            print_population(self._population)
        if live_plot:
            plt.show()
            axes = plt.gca()
            self._xdata = []
            self._ydata = []
            axes.set_xlim(0, self._max_num_generations)
            max_fitness = max([x.get_fitness() for x in self._population])
            axes.set_ylim(0, int(max_fitness * 0.1))
            self._line, = axes.plot(self._xdata, self._ydata, '-')

        self._num_generations = 0

    def next_generation():

        if self._num_generations >= self._max_num_generations:
            print("done")
            return self._population

        print("generation:", self._num_generations, "last fittest:", self._population[0].get_fitness(), "genom:", self._population[0].get_genom())
        if live_plot:
            self._xdata.append(self._num_generations)
            self._ydata.append(self._population[0].get_fitness())
            self._line.set_xdata(self._xdata)
            self._line.set_ydata(self._ydata)
            plt.draw()
            plt.pause(1e-17)

        # sort by fitness
        population = self._sort_by_fitness(population)

        # select the best-fit individuals for reproduction. (Parents)
        # we take 25 % of all best individuals
        parents = population[0:population_size // 4]

        # breed new individuals through crossover and mutation operations to give birth to offspring.
        childs = crossover(parents, mutation_rate)

        # Evaluate the individual fitness of new individuals.
        calc_fitness_of_all(childs, pool)

        population = population + childs
        population = sort_by_fitness(population)

        # replace least-fit population with new individuals.
        population = population[0:population_size]
        num_generations += 1

        print(population[0].print())
        print(population[0].get_genom())
        return population

    def _calc_fitness_of_all(self, population):
        # calculate fitness, parallel
        r = self._pool.map(_calc_fitness, population)
        for i, x in enumerate(r):
            population[i]._fitness = x
        return population

    def _print_population(self, population):
        print("\n".join(map(str, population)))

    def _print_genes(self, population):
        print("\n".join(map(lambda x: str(x.get_genom()), population)))

    def _crossover(self, parents, mutation_rate=0.5):
        childs = []
        while len(childs) < 2 * len(parents):
            # grab random parents and "merge them" to get a child
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = p1.sex(p2)
            child.mutate(mutation_rate)
            childs.append(child)

        return childs

    def _sort_by_fitness(self, population):
        return sorted(population, key=lambda x: x.get_fitness())




def main(_):
    random.seed(42)
    # initialization of initial population
    population_size = 200
    max_num_generations = 1000
    mutation_rate = 0.1
    INDIVIDUAL_CLASS = HelloWorld
    cpu_count = 4






if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))