import random
import numpy as np
import copy
import pandas as pd
import ast
import operator
import tensorflow as tf

from nnga.genetic_algorithm.population import Population
from nnga.genetic_algorithm import set_parameters
from nnga import get_architecture
from nnga.utils.data_io import (
    load_statistic,
    load_pop,
    save_statistic,
    save_pop,
)
from nnga.model_training import ModelTraining
from nnga.utils.helper import dump_tensors
from nnga.utils.data_io import save_feature_selected


class GA:
    """ This class implements the Genetic Algorithm
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            datasets: BaseDataset
                dataset to be used in the train/test
        Returns
        -------
    """

    def __init__(self, cfg, logger, datasets):

        self.datasets = datasets
        self._cfg = cfg

        self.nrMaxGen = cfg.GA.NRO_MAX_GEN
        self.population_size = cfg.GA.POPULATION_SIZE
        self.crossoverRate = cfg.GA.CROSSOVER_RATE
        self.mutationRate = cfg.GA.MUTATION_RATE

        self.elitism = cfg.GA.ELITISM
        self.tournament_size = cfg.GA.TOURNAMENT_SIZE

        self.hypermutation = cfg.GA.HYPERMUTATION
        self.cycleSize = cfg.GA.HYPERMUTATION_CYCLE_SIZE

        self.hypermutationRate = cfg.GA.HYPERMUTATION_RATE
        self._hypermutationCycle = False

        self.randomImmigrants = cfg.GA.RANDOM_IMIGRANTS
        self.immigrationRate = cfg.GA.IMIGRATION_RATE

        self._statistic = []

        self.continue_exec = cfg.GA.CONTINUE_EXEC
        self.feature_selection = cfg.MODEL.FEATURE_SELECTION

        self._seed = cfg.GA.SEED

        self._path = cfg.OUTPUT_DIR
        self._logger = logger

        # Check variables

        if not isinstance(self.nrMaxGen, int) or self.nrMaxGen <= 0:
            raise ValueError(
                "GA.NRO_MAX_GEN must be a int number " "(0 < GA.NRO_MAX_GEN)"
            )

        if (
            not isinstance(self.population_size, int)
            or self.population_size <= 0
        ):
            raise ValueError(
                "GA.POPULATION_SIZE must be a int number "
                "(0 < GA.POPULATION_SIZE)"
            )

        if (
            not isinstance(self.crossoverRate, float)
            or self.crossoverRate <= 0
            or self.crossoverRate >= 1
        ):
            raise ValueError(
                "GA.CROSSOVER_RATE must be a float number "
                "(0 < GA.CROSSOVER_RATE < 1)"
            )

        if (
            not isinstance(self.mutationRate, float)
            or self.mutationRate <= 0
            or self.mutationRate >= 1
        ):
            raise ValueError(
                "GA.MUTATION_RATE must be a float number "
                "(0 < GA.MUTATION_RATE < 1)"
            )

        if (
            not isinstance(self.elitism, int)
            or self.elitism <= 0
            or self.elitism > self.population_size
        ):
            raise ValueError(
                "GA.ELITISM must be a int number "
                "(0 < GA.ELITISM < GA.POPULATION_SIZE)"
            )

        if (
            not isinstance(self.tournament_size, int)
            or self.tournament_size <= 0
            or self.tournament_size > self.population_size
        ):
            raise ValueError(
                "GA.TOURNAMENT_SIZE must be a int number "
                "(0 < GA.TOURNAMENT_SIZE < GA.POPULATION_SIZE)"
            )

        if not isinstance(self.hypermutation, bool):
            raise ValueError("GA.HYPERMUTATION must be a bool")

        if (
            not isinstance(self.cycleSize, int)
            or self.cycleSize <= 0
            or self.cycleSize > self.nrMaxGen
        ):
            raise ValueError(
                "GA.HYPERMUTATION_CYCLE_SIZE must be "
                "a int number "
                "(0 < GA.HYPERMUTATION_CYCLE_SIZE < "
                "GA.NRO_MAX_GEN)"
            )

        if (
            not (
                isinstance(self.hypermutationRate, float)
                or isinstance(self.hypermutationRate, int)
            )
            or self.hypermutationRate <= 0
        ):
            raise ValueError(
                "GA.HYPERMUTATION_RATE must be a int "
                "or float number "
                "(0 < GA.HYPERMUTATION_RATE)"
            )

        if not isinstance(self.randomImmigrants, bool):
            raise ValueError("GA.RANDOM_IMIGRANTS must be a bool")

        if (
            not isinstance(self.immigrationRate, float)
            or self.immigrationRate <= 0
            or self.immigrationRate >= 1
        ):
            raise ValueError(
                "GA.IMIGRATION_RATE must be a float number "
                "(0 < GA.IMIGRATION_RATE < 1)"
            )

        if not isinstance(self._seed, int):
            raise ValueError("GA.SEED must be a int number")

        if not isinstance(self.continue_exec, bool):
            raise ValueError("GA.CONTINUE_EXEC must be a bool")

        if not isinstance(self.continue_exec, bool):
            raise ValueError("GA.CONTINUE_EXEC must be a bool")

        if not isinstance(self.feature_selection, bool):
            raise ValueError("MODEL.FEATURE_SELECTION must be a bool")

        self.old_pop = Population(self.population_size)
        self.new_pop = Population(self.population_size)

        if self.feature_selection and "MLP" in cfg.MODEL.ARCHITECTURE:
            self._features = self.datasets["TRAIN"].features
        else:
            self._features = []
            self.feature_selection = False

        (
            self._encoding,
            self._name_features,
            self._parameters_interval,
        ) = set_parameters(
            cfg.GA.SEARCH_SPACE,
            cfg.MODEL.ARCHITECTURE,
            cfg.MODEL.BACKBONE,
            self._features,
        )

        self._encoding_keys = list(self._encoding.keys())

        self.chromosome_length = len(self._encoding)

        # self.mutationRate *= self.chromosome_length

        self._logger.info(f"GA created!")
        self._logger.info(f"Chromosome_lenght: {self.chromosome_length}")
        # self._logger.info(f"GA created \n"
        #                   f"{vars(self)}")

    def _generate_initial_population(self):
        """Generate a initical population"""
        for i in range(self.population_size):
            self.new_pop.indivs[i].chromosome = self._generating_indiv()
            self.new_pop.indivs[i].fitness = self._calculate_fitness(
                self.new_pop.indivs[i].chromosome, i, 0
            )

    def _starting_population(self):
        """ Start the population
            Generate a new pop or read from file
        """
        # Read pop to continue the exec
        if self.continue_exec:

            statistic = load_statistic(self._path)
            last_pop = load_pop(self._path)
            gen_init = statistic.shape[0]

            for i in range(gen_init):
                self._statistic.append(
                    {
                        "mean_fitness": statistic["mean_fitness"][i],
                        "best_fitness": statistic["best_fitness"][i],
                        "best_indiv": statistic["best_indiv"][i],
                    }
                )

            for i in range(self.population_size):
                self.new_pop.indivs[i].chromosome = np.array(
                    ast.literal_eval(last_pop["indiv"][i].replace(" ", ", ")),
                    dtype=object,
                )
                self.new_pop.indivs[i].fitness = last_pop["fitness"][i]

        # Starting a new pop
        else:
            self._generate_initial_population()
            gen_init = 1

        self._population_statistic()
        self._print_population_info(self.new_pop, gen_init - 1)

        return gen_init

    def _population_statistic(self):
        """ Calculate the population statistic"""
        self.new_pop.bestIndiv = []
        self.new_pop.sumFitness = 0

        for i in range(0, self.population_size):
            self.new_pop.sumFitness = (
                self.new_pop.sumFitness + self.new_pop.indivs[i].fitness
            )
            self.new_pop.bestIndiv.append((i, self.new_pop.indivs[i].fitness))

        self.new_pop.bestIndiv.sort(key=operator.itemgetter(1), reverse=True)
        self.new_pop.bestIndiv = self.new_pop.bestIndiv[
            0 : self.elitism if self.elitism != 0 else 2
        ]
        self.new_pop.bestIndiv = [t[0] for t in self.new_pop.bestIndiv]

        self.new_pop.maxFitness = self.new_pop.indivs[
            self.new_pop.bestIndiv[0]
        ].fitness
        self.new_pop.meanFitness = (
            self.new_pop.sumFitness / self.population_size
        )

        self._statistic.append(
            {
                "mean_fitness": self.new_pop.meanFitness,
                "best_fitness": self.new_pop.maxFitness,
                "best_indiv": self.new_pop.indivs[
                    self.new_pop.bestIndiv[0]
                ].chromosome,
            }
        )

    def _select_individual_by_tournament(self, pop):
        """ Start the population
            Generate a new pop or read from file
            Parameters
            ----------
                pop: Population
                    Pop to be used to select the individual by tournament
            Returns
            -------
        """

        # Pick individuals for tournament
        fighters = random.sample(
            range(0, self.population_size), self.tournament_size
        )

        # Identify individual with highest fitness
        winner = fighters[0]
        winner_fitness = pop[fighters[0]].fitness
        for fighter in fighters:
            if pop[fighter].fitness > winner_fitness:
                winner = fighter
                winner_fitness = pop[fighter].fitness

        return winner

    def _generating_new_population(self, gen):
        """ Generate the new pop from next generation
            Parameters
            ----------
                gen: int
                    Gen number
            Returns
            -------
        """
        self.old_pop = copy.deepcopy(self.new_pop)

        if self.hypermutation and gen % self.cycleSize == 0:
            self._hypermutation(gen)

        if self.elitism > 0:
            for i in range(self.elitism):
                self.new_pop.indivs[i] = self.old_pop.indivs[
                    self.old_pop.bestIndiv[i]
                ]
                # self.new_pop.indivs[i].fitness = self._calculate_fitness(
                #     self.new_pop.indivs[i].chromosome, i, gen
                # )
                self.new_pop.indivs[i].fitness = self.old_pop.indivs[
                    self.old_pop.bestIndiv[i]
                ].fitness

        for i in range(self.elitism, self.population_size, 2):
            parent_1 = self._select_individual_by_tournament(
                self.old_pop.indivs
            )
            parent_2 = self._select_individual_by_tournament(
                self.old_pop.indivs
            )

            # Crossover
            child_1, child_2 = self._crossover(
                self.old_pop.indivs[parent_1].chromosome,
                self.old_pop.indivs[parent_2].chromosome,
            )

            # Mutation
            self._randomly_mutate(child_1)
            self._randomly_mutate(child_2)

            # child_1
            self.new_pop.indivs[i].chromosome = child_1
            self.new_pop.indivs[i].fitness = self._calculate_fitness(
                child_1, i, gen
            )
            self.new_pop.indivs[i].parent_1 = parent_1
            self.new_pop.indivs[i].parent_2 = parent_2

            # child_2
            if i + 1 < self.population_size:
                self.new_pop.indivs[i + 1].chromosome = child_2
                self.new_pop.indivs[i + 1].fitness = self._calculate_fitness(
                    child_2, i + 1, gen
                )
                self.new_pop.indivs[i + 1].parent_1 = parent_1
                self.new_pop.indivs[i + 1].parent_2 = parent_2

        if self.randomImmigrants:
            for i in range(self.elitism, self.population_size):
                if random.random() < self.immigrationRate:
                    self.new_pop.indivs[
                        i
                    ].chromosome = self._generating_indiv()
                    self.new_pop.indivs[i].fitness = self._calculate_fitness(
                        self.new_pop.indivs[i].chromosome, i, gen
                    )

    def _crossover(self, parent_1, parent_2):
        """ Generate the new pop from next generation
            Parameters
            ----------
                parent_1: Indiv
                    Indiv to be used to create a new Indiv by crossover
                parent_2: Indiv
                Indiv to be used to create a new Indiv by crossover
            Returns
            -------
                Return two new Indivs
        """
        # Pick crossover points
        crossover_points = random.sample(range(0, self.chromosome_length), 2)

        # two points crossover
        if crossover_points[0] != crossover_points[1]:

            # Order by the points. Smaller first
            if crossover_points[0] > crossover_points[1]:
                aux = crossover_points[0]
                crossover_points[0] = crossover_points[1]
                crossover_points[1] = aux

            child_1 = np.hstack(
                (
                    parent_1[0 : crossover_points[0]],
                    parent_2[crossover_points[0] : crossover_points[1]],
                    parent_1[crossover_points[1] :],
                )
            )

            child_2 = np.hstack(
                (
                    parent_2[0 : crossover_points[0]],
                    parent_1[crossover_points[0] : crossover_points[1]],
                    parent_2[crossover_points[1] :],
                )
            )

        # one point crossover
        else:
            child_1 = np.hstack(
                (
                    parent_1[0 : crossover_points[0]],
                    parent_2[crossover_points[0] :],
                )
            )

            child_2 = np.hstack(
                (
                    parent_2[0 : crossover_points[0]],
                    parent_1[crossover_points[0] :],
                )
            )

        return child_1, child_2

    def run(self):
        """Run the Genetic Algorithm"""
        self._logger.info(
            f"\n\n*************************  "
            f"Genetic Algorithm   "
            f"*************************"
        )

        self._logger.info(
            f"============================ Execution with seed: "
            f"{self._seed}  ============================"
        )

        random.seed(self._seed)
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)

        gen_init = self._starting_population()

        for gen in range(gen_init, self.nrMaxGen + 1):
            self._generating_new_population(gen)
            self._population_statistic()
            self._print_population_info(self.new_pop, gen)

            save_pop(self._path, self.new_pop)
            save_statistic(self._path, self._statistic)

        self._evaluate_ga_results(
            indiv=self.new_pop.indivs[self.new_pop.bestIndiv[0]].chromosome,
        )

    def _print_population_info(self, pop, gen):
        """Print population info
        Parameters
            ----------
                pop: Population
                    Pop to be print info
                gen: int
                    Gen number
            Returns
            -------
        """
        self._logger.info(
            f"Generation: {gen}\n"
            f"Fitness best indiv: {pop.maxFitness}\n"
            f"Best indivs: {pop.bestIndiv}\n"
            f"Fitness mean: {pop.meanFitness}\n"
            f"Mutation rate: {self.mutationRate}\n\n"
        )

    def _hypermutation(self, gen):
        """Hypermutation: change the mutation rate if necessary
        Parameters
            ----------
                gen: int
                    Gen number
            Returns
            -------
        """
        if self._hypermutationCycle:
            self.mutationRate = self.mutationRate / self.hypermutationRate
            self._hypermutationCycle = False
        elif gen % (2 * self.cycleSize) == 0:

            df = pd.DataFrame(
                self._statistic,
                columns=["mean_fitness", "best_fitness", "best_indiv"],
            )
            cycle_1 = sum(
                df["best_fitness"][
                    gen - 2 * self.cycleSize : gen - self.cycleSize
                ]
            )
            cycle_2 = sum(df["best_fitness"][gen - self.cycleSize : gen])

            if cycle_1 >= cycle_2:
                self.mutationRate = self.mutationRate * self.hypermutationRate
                self._hypermutationCycle = True

    def _generating_indiv_binary(self):
        """Generate new indiv binary
        Parameters
            ----------
            Returns
            -------
                New indiv chromosome
        """
        # Set up an initial array of all zeros
        chromosome = np.zeros(self.chromosome_length)

        # Choose a random number of ones to create
        ones = random.randint(0, self.chromosome_length)
        # Change the required number of zeros to ones
        chromosome[0:ones] = 1
        # Sfuffle row
        np.random.shuffle(chromosome)

        return chromosome

    def _generating_indiv(self):
        """Generate new indiv custom
        Parameters
            ----------
            Returns
            -------
                New indiv chromosome
        """
        chromosome = []

        for key in self._encoding_keys:
            chromosome.append(
                random.choice(
                    self._parameters_interval[self._encoding[key]]["value"]
                )
            )

        return np.array(chromosome, dtype=object)

    def _randomly_mutate_binary(self, chromosome):
        """Randomly mutate indiv binary
        Parameters
            ----------
                chromosome: list
                    Indiv chromosome
            Returns
            -------
        """
        for i in range(self.chromosome_length):
            if random.random() < self.mutationRate:
                if chromosome[i] == 1:
                    chromosome[i] = 0
                else:
                    chromosome[i] = 1

    def _randomly_mutate(self, chromosome):
        """Randomly mutate indiv custom
            Parameters
            ----------
                chromosome: list
                    Indiv chromosome
            Returns
            -------
        """

        for i in range(self.chromosome_length):
            if random.random() < self.mutationRate:
                options = self._parameters_interval[
                    self._encoding[self._encoding_keys[i]]
                ]["value"].copy()

                # Create a index of windows
                windows_min = (
                    options.index(chromosome[i]) - 2
                    if options.index(chromosome[i]) - 2 >= 0
                    else 0
                )
                windows_max = (
                    options.index(chromosome[i]) + 3
                    if options.index(chromosome[i]) + 3 <= len(options)
                    else len(options)
                )

                options = options[windows_min:windows_max]
                if len(options) > 1:
                    options.remove(chromosome[i])
                    chromosome[i] = random.choice(options)

    def _calculate_fitness(self, indiv, i, gen):
        """Calculate fitness from indiv
        Parameters
            ----------
                indiv: Indiv
                    Indiv to calculate the fitness
                i: int
                    id of the individual in the population
                gen: int
                    Gen number to determinate the
                    train_test_split random_state
            Returns
            -------
        """

        self._logger.info(
            f"Indiv {i}: {dict(zip(self._encoding_keys, indiv))}"
        )

        if self.feature_selection:
            features_idx, features_name = self.idx_features_selected(indiv)
            self._logger.info(
                f"Features selected {len(features_name)}: {features_name}"
            )
        else:
            features_idx = None

        MakeModel = get_architecture(self._cfg.TASK, self._cfg.MODEL.ARCHITECTURE)

        try:
            model = MakeModel(
                self._cfg,
                self._logger,
                self.datasets["TRAIN"].input_shape,
                self.datasets["TRAIN"].n_classes,
                indiv=indiv,
                keys=self._encoding_keys,
            )
            model_trainner = ModelTraining(
                self._cfg,
                model,
                self._logger,
                self.datasets,
                indiv,
                self._encoding_keys,
                features_idx,
            )

            evaluate = model_trainner.train_test_split(random_state=0)
            metrics = model_trainner.compute_metrics()
            fitness = float(metrics["balanced_accuracy_score"])
            # fitness = 1 / (1 + evaluate[0])

            self._logger.info(
                f"balanced accuracy: {metrics['balanced_accuracy_score']}"
            )
            self._logger.info(
                f"confusion matrix: \n{metrics['confusion_matrix']}"
            )
        except ValueError:
            evaluate = [float("inf"), 1e-5]
            fitness = 0.0
        except Exception as e:
            self._logger.error(e)
            raise e

        self._logger.info(
            f"evaluate (loss value & metrics values): {evaluate}"
        )
        self._logger.info(f"Fitness: {fitness}\n")

        dump_tensors()

        return fitness

    def _evaluate_ga_results(self, indiv):
        """Calculate fitness from indiv
        Parameters
            ----------
                indiv: Indiv
                    Indiv to calculate the fitness
            Returns
            -------
        """
        self._logger.info(f"Indiv: {dict(zip(self._encoding_keys, indiv))}")

        if self.feature_selection:
            features_idx, features_name = self.idx_features_selected(indiv)
            self._logger.info(
                f"Features selected {len(features_name)}: {features_name}"
            )
            save_feature_selected(self._path, features_name)
        else:
            features_idx = None

        MakeModel = get_architecture(self._cfg.TASK, self._cfg.MODEL.ARCHITECTURE)

        try:
            model = MakeModel(
                self._cfg,
                self._logger,
                self.datasets["TRAIN"].input_shape,
                self.datasets["TRAIN"].n_classes,
                indiv=indiv,
                keys=self._encoding_keys,
            )

            model_trainner = ModelTraining(
                self._cfg,
                model,
                self._logger,
                self.datasets,
                indiv,
                self._encoding_keys,
                features_idx,
            )

            if self._cfg.SOLVER.CROSS_VALIDATION:
                cv = model_trainner.cross_validation(
                    random_state=self._seed, save=True
                )
                self._logger.info(f"Cross validation statistics:\n{cv}")

            model_trainner.fit()
            evaluate = model_trainner.evaluate()
            # fitness = 1 / (1 + evaluate[0])
            metrics = model_trainner.compute_metrics(save=True)
            fitness = float(metrics["balanced_accuracy_score"])

            self._logger.info(
                f"balanced accuracy: {metrics['balanced_accuracy_score']}"
            )
            self._logger.info(
                f"confusion matrix: \n{metrics['confusion_matrix']}"
            )

            model.save_model()
        except ValueError:
            evaluate = [float("inf"), 1e-5]
            fitness = 0.0
        except Exception as e:
            self._logger.error(e)
            raise e

        self._logger.info(
            f"evaluate (loss value & metrics values): {evaluate}"
        )
        self._logger.info(f"Fitness solution: {fitness}")

        dump_tensors()

    def idx_features_selected(self, indiv):
        """Get idxs and names of feature selected
        from indiv
        Parameters
        ----------
            indiv: Indiv
                Indiv to calculate the fitness
        Returns
        -------
            (idx_features_selected, name_features_selected): (list, list)
                idx_features_selected: idx from features selected by GA
                name_features_selected: name from features selected by GA
        """
        indexes = [
            i
            for i, value in enumerate(self._encoding_keys)
            if "feature_selection_" in value
        ]

        idx_features_selected = [
            i for i, value in enumerate(indexes) if indiv[value]
        ]

        name_features_selected = [
            self._name_features[self._encoding_keys[i]]
            for i in indexes
            if indiv[i]
        ]

        return idx_features_selected, name_features_selected
