#!/usr/bin/env python3
"""
Genetic evoltion experiments, uses multiprocessing
"""
"""
    This file is part of quat.
    quat is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    quat is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with quat. If not, see <http://www.gnu.org/licenses/>.

    Author: Steve Göring
"""
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
    """
    base class of an individual in a genetic algorithm
    """
    _genom = np.array([])
    _fitness = 0

    def __init__(self, number_genes=42):
        self._genom = np.array([0.0] * number_genes)

    def init_genom(self):
        """
        inititalize genom of individual, with random numbers, [0,1] per default
        """
        for i in range(len(self._genom)):
            self._genom[i] = random.uniform(0, 1)
        return self  # return self, simplifies construction

    def get_genom(self):
        """
        return genom
        """
        return self._genom

    def str_genom(self):
        """
        string version of genom
        """
        return str(self.get_genom())

    def __str__(self):
        return f"""{self.__class__.__name__}: fit: {self._fitness} """

    def print(self):
        return str(self)

    def calc_fitness(self):
        """
        calculate fitness of individual
        """
        raise NotImplementedError("this method should be implemented by your subclass")

    def get_fitness(self):
        """
        return fitness value
        """
        return self._fitness

    def crossover(self, other):
        """
        combine current individual with other,
        create a new individual (copy)
        """
        c = copy.deepcopy(self)
        c._fitness = 0
        g = c.get_genom()
        go = other.get_genom()
        split = int(random.random() * len(g))
        c._genom = np.concatenate([g[0:split], go[split:]])
        return c

    def mutate(self, mutation_rate):
        """
        based on mutation rate, change some genes,
        mutation rate is the percentage of how many genes are changes
        (randomly choosen, with repetition)
        """
        for x in range(0, max(1, int(np.ceil(mutation_rate * len(self._genom))))):
            pos = random.randint(0, len(self._genom) - 1)
            self._genom[pos] = np.clip(self._genom[pos] +  (2 * random.random() - 1), 0, 1)
        self._fitness = 0



class HelloWorld(Individual):
    """
    example individual class, starting from a random string,
    get "HelloWorld"
    """
    target = np.array([ord(x) - 65 for x in "HelloWorld"], dtype=np.int)

    def __init__(self, number_genes=len("HelloWorld")):
        super().__init__(number_genes)

    def calc_fitness(self):
        """
        in this case fitness is the difference to the target string
        """
        nums = [int((122 - 65) * x) for x in self._genom]
        diff = (np.array(nums, dtype=np.int) - self.target) ** 2
        fit = diff.sum()
        self._fitness = fit
        return fit

    def str_genom(self):
        return self.get_genom_as_char()

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
    """ local helper for multiprocessing
    """
    individual.calc_fitness()
    return individual.get_fitness()


class GeneticEvolution:
    """
    simple base class for genetic evolution
    """
    def __init__(self, population_size, mutation_rate, max_num_generations, individual_class, cpu_count=multiprocessing.cpu_count(), verbose=False, live_plot=True, checkpoint_folder="checkpoint", checkpoint_intervall=10):
        """
        create GeneticEvolution instance
        """
        # create multiprocessing pool
        self._pool = Pool(processes=cpu_count)
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._max_num_generations = max_num_generations
        self._individual_class = individual_class
        self._cpu_count = cpu_count
        self._verbose = verbose
        self._live_plot = live_plot
        self._checkpoint_folder = checkpoint_folder
        self._checkpoint_intervall = checkpoint_intervall
        os.makedirs(self._checkpoint_folder, exist_ok=True)

        # create initial population
        self._population = []
        for i in range(self._population_size):
            self._population.append(self._individual_class().init_genom())
        self._population = np.array(self._population)

        self._calc_fitness_of_all(self._population)
        if self._verbose:
            self._print_population(self._population)
        if self._live_plot:
            plt.show()
            axes = plt.gca()
            self._xdata = []
            self._ydata = []
            axes.set_xlim(0, self._max_num_generations)
            max_fitness = max([x.get_fitness() for x in self._population])
            axes.set_ylim(0, int(max_fitness * 0.1))
            self._line, = axes.plot(self._xdata, self._ydata, '-')

        self._num_generations = 0

    def next_generation(self):
        """
        evolve the next generation
        """
        if self._num_generations >= self._max_num_generations:
            print("done")
            return self._population

        print("generation:", self._num_generations, "last fittest:", self._population[0].get_fitness(), "genom:", self._population[0].str_genom())
        if self._live_plot:
            self._xdata.append(self._num_generations)
            self._ydata.append(self._population[0].get_fitness())
            self._line.set_xdata(self._xdata)
            self._line.set_ydata(self._ydata)
            plt.draw()
            plt.pause(1e-17)

        # sort by fitness
        self._population = self._sort_by_fitness(self._population)

        # select the best-fit individuals for reproduction. (Parents)
        # we take 25 % of all best individuals
        parents = self._population[0:self._population_size // 4]

        # breed new individuals through crossover and mutation operations to give birth to offspring.
        childs = self._crossover(parents, self._mutation_rate)

        # Evaluate the individual fitness of new individuals.
        self._calc_fitness_of_all(childs)

        self._population = self._population + childs
        self._population = self._sort_by_fitness(self._population)

        # replace least-fit population with new individuals.
        self._population = self._population[0:self._population_size]
        self._num_generations += 1

        if self._verbose:
            print(self._population[0].print())
            print(self._population[0].str_genom())

        self._checkpoint()
        return self._population

    def _checkpoint(self):
        if self._num_generations % self._checkpoint_intervall == 0:
            print(f"checkpoint: {self._checkpoint_folder}/{self._num_generations}")
            # TODO

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
            child = p1.crossover(p2)
            child.mutate(mutation_rate)
            childs.append(child)

        return childs

    def _sort_by_fitness(self, population):
        return sorted(population, key=lambda x: x.get_fitness())

    def get_number_generations(self):
        return self._num_generations


def main(_):
    """
    example usage of the GeneticEvolution classes
    """
    random.seed(42)

    # define parameters
    population_size = 200
    max_num_generations = 1000
    mutation_rate = 0.1
    cpu_count = 4

    # create instance
    ga = GeneticEvolution(
        population_size,
        mutation_rate,
        max_num_generations,
        individual_class=HelloWorld,
        cpu_count=multiprocessing.cpu_count(),
        verbose=True,
        live_plot=True,
        checkpoint_folder="checkpoints"
    )

    # let evolution run as long as we need it
    while ga.get_number_generations() < 500:
        ga.next_generation()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
