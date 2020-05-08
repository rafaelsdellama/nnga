from nnga.genetic_algorithm.indiv import Indiv


def test_indiv():
    indiv = Indiv()
    assert type(indiv) == Indiv
