from nnga.genetic_algorithm.indiv import Indiv


class Population:

    def __init__(self, population_size):
        self.indivs = [Indiv() for i in range(population_size)]
        self.sumFitness = None
        self.meanFitness = None
        self.maxFitness = None
        self.bestIndivs = None
