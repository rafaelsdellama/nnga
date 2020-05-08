from nnga.genetic_algorithm.indiv import Indiv


class Population:
    """ This class implements the Population from genetic algorithm
        Parameters
        ----------
        population_size: int
            Population size

        Returns
        -------
    """

    def __init__(self, population_size):
        self.indivs = [Indiv() for i in range(population_size)]
        self.sumFitness = None
        self.meanFitness = None
        self.maxFitness = None
        self.bestIndivs = None

    def __len__(self):
        return len(self.indivs)
