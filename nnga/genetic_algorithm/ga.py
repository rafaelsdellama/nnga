import random
import numpy as np
import copy
import pandas as pd
import ast
import operator
import tensorflow as tf

from nnga.genetic_algorithm.population import Population
from nnga.genetic_algorithm import ENCODING, PARAMETERS_INTERVAL, MODELS
from nnga.utils.data_io import load_statistic, load_pop, \
    save_statistic, save_pop
from nnga.utils.helper import dump_tensors


class GA:
    """Genetic Algorithm class"""

    def __init__(self, cfg, logger, datasets):
        """
        Create datasets, Models following experiment config
            Arguments:
                cfg {yacs.config.CfgNode} -- Loaded experiment config
                logger {logging} -- Simple python logging
        """

        self.datasets = datasets
        self._cfg = cfg
        self.encoding = ENCODING.get(cfg.MODEL.ARCHITECTURE)

        self.nrMaxExec = cfg.GA.NRO_MAX_EXEC
        self.nrMaxGen = cfg.GA.NRO_MAX_GEN
        self.population_size = cfg.GA.POPULATION_SIZE
        self.crossoverRate = cfg.GA.CROSSOVER_RATE
        self.mutationRate = cfg.GA.MUTATION_RATE

        self.elitism = cfg.GA.ELITISM
        self.tournament_size = cfg.GA.TOURNAMENT_SIZE

        self.hypermutation = cfg.GA.HYPERMUTATION
        self.cycleSize = cfg.GA.HYPERMUTATION_CYCLE_SIZE

        self.hypermutationRate = cfg.GA.HYPERMUTATION_RATE
        self.__hypermutationCycle = False

        self.randomImmigrants = cfg.GA.RANDOM_IMIGRANTS
        self.immigrationRate = cfg.GA.IMIGRATION_RATE

        self.old_pop = Population(self.population_size)
        self.new_pop = Population(self.population_size)
        self.__statistic = []

        if len(cfg.GA.SEED) == 0:
            self.__seeds = list(range(self.nrMaxExec))
        else:
            self.__seeds = cfg.GA.SEED

        self.continue_exec = cfg.GA.CONTINUE_EXEC
        self.feature_selection = cfg.GA.FEATURE_SELECTION

        self._path = cfg.OUTPUT_DIR
        self._logger = logger

        # Check variables
        if not isinstance(self.nrMaxExec, int) or self.nrMaxExec <= 0:
            raise ValueError("GA.NRO_MAX_EXEC must be a int number "
                             "(0 < GA.NRO_MAX_EXEC)")

        if not isinstance(self.nrMaxGen, int) or self.nrMaxGen <= 0:
            raise ValueError("GA.NRO_MAX_GEN must be a int number "
                             "(0 < GA.NRO_MAX_GEN)")

        if not isinstance(self.population_size, int) or \
                self.population_size <= 0:
            raise ValueError("GA.POPULATION_SIZE must be a int number "
                             "(0 < GA.POPULATION_SIZE)")

        if not isinstance(self.crossoverRate, float) or \
                self.crossoverRate <= 0 or self.crossoverRate >= 1:
            raise ValueError("GA.CROSSOVER_RATE must be a float number "
                             "(0 < GA.CROSSOVER_RATE < 1)")

        if not isinstance(self.mutationRate, float) or \
                self.mutationRate <= 0 or self.mutationRate >= 1:
            raise ValueError("GA.MUTATION_RATE must be a float number "
                             "(0 < GA.MUTATION_RATE < 1)")

        if not isinstance(self.elitism, int) or \
                self.elitism <= 0 or \
                self.elitism > self.population_size:
            raise ValueError("GA.ELITISM must be a int number "
                             "(0 < GA.ELITISM < GA.POPULATION_SIZE)")

        if not isinstance(self.tournament_size, int) or \
                self.tournament_size <= 0 or \
                self.tournament_size > self.population_size:
            raise ValueError("GA.TOURNAMENT_SIZE must be a int number "
                             "(0 < GA.TOURNAMENT_SIZE < GA.POPULATION_SIZE)")

        if not isinstance(self.hypermutation, bool):
            raise ValueError("GA.HYPERMUTATION must be a bool")

        if not isinstance(self.cycleSize, int) or \
                self.cycleSize <= 0 or \
                self.cycleSize > self.nrMaxGen:
            raise ValueError("GA.HYPERMUTATION_CYCLE_SIZE must be "
                             "a int number "
                             "(0 < GA.HYPERMUTATION_CYCLE_SIZE < "
                             "GA.NRO_MAX_GEN)")

        if not (isinstance(self.hypermutationRate, float) or
                isinstance(self.hypermutationRate, int)) or \
                self.hypermutationRate <= 0:
            raise ValueError("GA.HYPERMUTATION_RATE must be a int "
                             "or float number "
                             "(0 < GA.HYPERMUTATION_RATE)")

        if not isinstance(self.randomImmigrants, bool):
            raise ValueError("GA.RANDOM_IMIGRANTS must be a bool")

        if not isinstance(self.immigrationRate, float) or \
                self.immigrationRate <= 0 or self.immigrationRate >= 1:
            raise ValueError("GA.IMIGRATION_RATE must be a float number "
                             "(0 < GA.IMIGRATION_RATE < 1)")

        if not isinstance(self.__seeds, list) or \
                not all(isinstance(x, int) for x in self.__seeds) or \
                len(self.__seeds) != self.nrMaxExec:
            raise ValueError("GA.SEED must be a list of numbers"
                             "(len(GA.IMIGRATION_RATE) = GA.NRO_MAX_EXEC)")

        if not isinstance(self.continue_exec, bool):
            raise ValueError("GA.CONTINUE_EXEC must be a bool")

        if not isinstance(self.continue_exec, bool):
            raise ValueError("GA.CONTINUE_EXEC must be a bool")

        if not isinstance(self.feature_selection, bool):
            raise ValueError("GA.FEATURE_SELECTION must be a bool")

        if self.feature_selection and 'CSV' in self.datasets['TRAIN']:
            for i in range(self.datasets['TRAIN']['CSV'].n_features):
                self.encoding[f"feature_selection_{i}"] = \
                    'feature_selection'

        self.chromosome_length = len(self.encoding)

        self._logger.info(f"GA created \n"
                          f"{vars(self)}")

    def __generate_initial_population(self):
        for i in range(self.population_size):
            self.new_pop.indivs[i].chromosome = self.__generating_indiv()
            self.new_pop.indivs[i].fitness = \
                self.__calculate_fitness(self.new_pop.indivs[i].chromosome, 0)

    def __starting_population(self, seed):
        # Read pop to continue the exec
        if self.continue_exec:

            statistic = load_statistic(self._path, seed)
            last_pop = load_pop(self._path, seed)
            gen_init = statistic.shape[0]

            for i in range(gen_init):
                self.__statistic.append(
                    {'mean_fitness': statistic['mean_fitness'][i],
                     'best_fitness': statistic['best_fitness'][i],
                     'best_indiv': statistic['best_indiv'][i]
                     })

            for i in range(self.population_size):
                self.new_pop.indivs[i].chromosome = np.array(ast.literal_eval(
                    last_pop['indiv'][i].replace(" ", ", ")), dtype=object)
                self.new_pop.indivs[i].fitness = last_pop['fitness'][i]

        # Starting a new pop
        else:
            self.__generate_initial_population()
            gen_init = 1

        self.__population_statistic()
        self.__print_population_info(self.new_pop, gen_init - 1)

        return gen_init

    def __population_statistic(self):
        self.new_pop.bestIndiv = []
        self.new_pop.sumFitness = 0

        for i in range(0, self.population_size):
            self.new_pop.sumFitness = \
                self.new_pop.sumFitness + self.new_pop.indivs[i].fitness
            self.new_pop.bestIndiv.append((i, self.new_pop.indivs[i].fitness))

        self.new_pop.bestIndiv.sort(
            key=operator.itemgetter(1),
            reverse=True
        )
        self.new_pop.bestIndiv = self.new_pop.bestIndiv[
                                 0: self.elitism if self.elitism != 0 else 2]
        self.new_pop.bestIndiv = [t[0] for t in self.new_pop.bestIndiv]

        self.new_pop.maxFitness = \
            self.new_pop.indivs[self.new_pop.bestIndiv[0]].fitness
        self.new_pop.meanFitness = \
            self.new_pop.sumFitness / self.population_size

        self.__statistic.append({'mean_fitness': self.new_pop.meanFitness,
                                 'best_fitness': self.new_pop.maxFitness,
                                 'best_indiv': self.new_pop.indivs[
                                     self.new_pop.bestIndiv[0]].chromosome})

    def __select_individual_by_tournament(self, pop):
        # Pick individuals for tournament
        fighters = random.sample(
            range(0, self.population_size),
            self.tournament_size
        )

        # Identify undividual with highest fitness
        winner = fighters[0]
        winner_fitness = pop[fighters[0]].fitness
        for fighter in fighters:
            if pop[fighter].fitness > winner_fitness:
                winner = fighter
                winner_fitness = pop[fighter].fitness

        return winner

    def __generating_new_population(self, gen):
        self.old_pop = copy.deepcopy(self.new_pop)

        if self.hypermutation and gen % self.cycleSize == 0:
            self.__hypermutation(gen)

        if self.elitism > 0:
            for i in range(self.elitism):
                self.new_pop.indivs[i] = \
                    self.old_pop.indivs[self.old_pop.bestIndiv[i]]
                self.new_pop.indivs[i].fitness = \
                    self.__calculate_fitness(
                        self.new_pop.indivs[0].chromosome,
                        gen
                    )

        for i in range(self.elitism, self.population_size, 2):
            parent_1 = self.__select_individual_by_tournament(
                self.old_pop.indivs
            )
            parent_2 = self.__select_individual_by_tournament(
                self.old_pop.indivs
            )

            # Crossover
            child_1, child_2 = self.__crossover(
                self.old_pop.indivs[parent_1].chromosome,
                self.old_pop.indivs[parent_2].chromosome)

            # Mutation
            self.__randomly_mutate(child_1)
            self.__randomly_mutate(child_2)

            # child_1
            self.new_pop.indivs[i].chromosome = child_1
            self.new_pop.indivs[i].fitness = \
                self.__calculate_fitness(
                    child_1,
                    gen
                )
            self.new_pop.indivs[i].parent_1 = parent_1
            self.new_pop.indivs[i].parent_2 = parent_2

            # child_2
            if i + 1 < self.population_size:
                self.new_pop.indivs[i + 1].chromosome = child_2
                self.new_pop.indivs[i + 1].fitness = \
                    self.__calculate_fitness(
                        child_2,
                        gen
                    )
                self.new_pop.indivs[i + 1].parent_1 = parent_1
                self.new_pop.indivs[i + 1].parent_2 = parent_2

        if self.randomImmigrants:
            for i in range(self.elitism, self.population_size):
                if random.random() < self.immigrationRate:
                    self.new_pop.indivs[i].chromosome = \
                        self.__generating_indiv()
                    self.new_pop.indivs[i].fitness = \
                        self.__calculate_fitness(
                            self.new_pop.indivs[i].chromosome,
                            gen
                        )

    def __crossover(self, parent_1, parent_2):
        # Pick crossover points
        crossover_points = random.sample(range(0, self.chromosome_length), 2)

        # two points crossover
        if crossover_points[0] != crossover_points[1]:

            # Order by the points. Smaller first
            if crossover_points[0] > crossover_points[1]:
                aux = crossover_points[0]
                crossover_points[0] = crossover_points[1]
                crossover_points[1] = aux

            child_1 = np.hstack((
                parent_1[0:crossover_points[0]],
                parent_2[crossover_points[0]:crossover_points[1]],
                parent_1[crossover_points[1]:]))

            child_2 = np.hstack((
                parent_2[0:crossover_points[0]],
                parent_1[crossover_points[0]:crossover_points[1]],
                parent_2[crossover_points[1]:]))

        # one point crossover
        else:
            child_1 = np.hstack((parent_1[0:crossover_points[0]],
                                 parent_2[crossover_points[0]:]))

            child_2 = np.hstack((parent_2[0:crossover_points[0]],
                                 parent_1[crossover_points[0]:]))

        return child_1, child_2

    def run(self):
        self._logger.info(f"\n\n*************************  "
                          f"Genetic Algorithm   "
                          f"*************************")

        for i, seed in enumerate(self.__seeds):
            self._logger.info(f"============================ Execution: "
                              f"{i + 1}/{self.nrMaxExec} - seed: {seed}  "
                              f"============================")

            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            self.__fit(seed)

            # Reset global variables
            self.__statistic = []

    def __fit(self, seed):

        gen_init = self.__starting_population(seed)

        for gen in range(gen_init, self.nrMaxGen + 1):
            self.__generating_new_population(gen)
            self.__population_statistic()
            self.__print_population_info(self.new_pop, gen)

            save_pop(self._path, seed, self.new_pop)
            save_statistic(self._path, seed, self.__statistic)

    def __print_population_info(self, pop, gen):
        self._logger.info(f"Generation: {gen}\n"
                          f"Fitness best indiv: {pop.maxFitness}\n"
                          f"Best indivs: {pop.bestIndiv}\n"
                          f"Fitness mean: {pop.meanFitness}\n"
                          f"Mutation rate: {self.mutationRate}\n\n")

    def __hypermutation(self, gen):
        if self.__hypermutationCycle:
            self.mutationRate = self.mutationRate / self.hypermutationRate
            self.__hypermutationCycle = False
        elif gen % (2 * self.cycleSize) == 0:

            df = pd.DataFrame(
                self.__statistic,
                columns=['mean_fitness', 'best_fitness', 'best_indiv']
            )
            cycle_1 = sum(df['best_fitness'][
                          gen - 2 * self.cycleSize:gen - self.cycleSize]
                          )
            cycle_2 = sum(df['best_fitness'][
                          gen - self.cycleSize:gen]
                          )

            if cycle_1 >= cycle_2:
                self.mutationRate = self.mutationRate * self.hypermutationRate
                self.__hypermutationCycle = True

    def __generating_indiv_binary(self):
        # Set up an initial array of all zeros
        chromosome = np.zeros(self.chromosome_length)

        # Choose a random number of ones to create
        ones = random.randint(0, self.chromosome_length)
        # Change the required number of zeros to ones
        chromosome[0:ones] = 1
        # Sfuffle row
        np.random.shuffle(chromosome)

        return chromosome

    def __generating_indiv(self):
        chromosome = []
        keys = list(self.encoding.keys())

        for key in keys:
            chromosome.append(
                random.choice(
                    PARAMETERS_INTERVAL[self.encoding[key]]))

        return np.array(chromosome, dtype=object)

    def __randomly_mutate_binary(self, chromosome):
        for i in range(self.chromosome_length):
            if random.random() < self.mutationRate:
                if chromosome[i] == 1:
                    chromosome[i] = 0
                else:
                    chromosome[i] = 1

    def __randomly_mutate(self, chromosome):
        keys = list(self.encoding.keys())

        for i in range(self.chromosome_length):
            if random.random() < self.mutationRate:
                options = PARAMETERS_INTERVAL[self.encoding[keys[i]]].copy()

                # Create a index of windows
                windows_min = options.index(chromosome[i]) - 2 \
                    if options.index(chromosome[i]) - 2 >= 0 \
                    else 0
                windows_max = options.index(chromosome[i]) + 3 \
                    if options.index(chromosome[i]) + 3 <= len(options) \
                    else len(options)

                options = options[windows_min:windows_max]
                options.remove(chromosome[i])
                chromosome[i] = random.choice(options)

    def __calculate_fitness(self, indiv, gen):
        keys = list(self.encoding.keys())
        self._logger.info(f"Indiv: {dict(zip(keys, indiv))}")
        model = MODELS.get(
            self._cfg.MODEL.ARCHITECTURE)(self._cfg,
                                          self._logger,
                                          self.datasets,
                                          indiv, keys,
                                          gen)

        if model.create_model_ga():
            model.fit()
            fitness = model.evaluate()
        else:
            fitness = 0.0

        del model
        dump_tensors()

        self._logger.info(f"Fitness: {fitness}\n")
        return fitness
