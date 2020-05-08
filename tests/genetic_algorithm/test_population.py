import pytest
from nnga.genetic_algorithm.population import Population


@pytest.mark.parametrize(
    "population_size", [0, 5, 10],
)
def test_population(population_size):
    pop = Population(population_size)
    assert len(pop) == population_size
