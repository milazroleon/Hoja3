from copy import deepcopy
from ga import GeneticSearch
import numpy as np
import time


def init(locations, random_state, n):

    """
    Creates an initial random population of size n for the TSP problem

    Args:
        locations (list): List of possible locations (just names/indices)
        random_state (np.random.RandomState): random state to control random behavior
        n (int): number of individuals in population

    Returns:
        list: a list of `n` individuals
    """

    population = []
    for _ in range(n):
        route = list(random_state.permutation(len(locations)))
        population.append(route)
    return population


def crossover(random_state, p1, p2):

    """
    Takes two individuals and combines them into two new routes.

    Args:
        random_state (np.random.RandomState): random state to control random behavior
        p1 (list): parent tour 1 (location indices)
        p2 (list): parent tour 2 (location indices)

    Returns:
        list: A list of size 2 with the offsprings of the parents p1 and p2 as entries, which are also lists themselves
    """

    n = len(p1)
    a, b = sorted(random_state.choice(n, size=2, replace=False))

    succ1 = [-1] * n
    succ2 = [-1] * n

    succ1[a:b] = p1[a:b]
    succ2[a:b] = p2[a:b]

    pos1, pos2 = b, b
    for gen in p2:
        if gen not in succ1:
            succ1[pos1 % n] = gen
            pos1 += 1
    for gen in p1:
        if gen not in succ2:
            succ2[pos2 % n] = gen
            pos2 += 1

    if np.array_equal(succ1, p1) or np.array_equal(succ1, p2):
        i, j = random_state.choice(n, size=2, replace=False)
        succ1[i], succ1[j] = succ1[j], succ1[i]
    if np.array_equal(succ2, p1) or np.array_equal(succ2, p2):
        i, j = random_state.choice(n, size=2, replace=False)
        succ2[i], succ2[j] = succ2[j], succ2[i]

    return [succ1, succ2]

    
def mutate(random_state, i):

    """
    Args:
        random_state (np.random.RandomState): random state to control random behavior
        i (list): tour to be mutated

    Returns:
        list: a mutant copy of the given individual `i`
    """

    m = list(i)
    a, b = random_state.choice(len(m), size=2, replace=False)
    m[a], m[b] = m[b], m[a]
    return m


def run_genetic_search_for_tsp(tsp, timeout):

    """
    
    Tries to find for (at most) timeout seconds to find a good solution for the TSP given by tsp.

    Args:
        tsp (TSP): The TSP to be solved
        timeout (int): Timeout in seconds after which a solution must have been returned (ideally earlier).

    Returns:
        list: The indices of the locations in the order that described the best tour that could be found.
    """

    inicio = time.time()
    rnd = np.random.RandomState()

    ciudades = list(range(len(tsp.locations)))
    tam_poblacion = min(50, len(ciudades) * 6)

    f_init = lambda n: init(ciudades, rnd, n)
    f_cross = lambda a, b: crossover(rnd, a, b)
    f_mut = lambda r: mutate(rnd, r)

    ga = GeneticSearch(f_init, f_cross, f_mut, tsp.is_better_route_than, tam_poblacion)
    ga.reset()

    best = deepcopy(ga.best)
    best_costo = tsp.get_cost_of_route(best)
    no_best = 0
    limite = 20 

    while time.time() - inicio < timeout:
        ga.step()
        candidato = ga.best
        costo_cand = tsp.get_cost_of_route(candidato)

        if costo_cand < best_costo:
            best = deepcopy(candidato)
            best_costo = costo_cand
            no_best = 0
        else:
            no_best += 1

        if no_best >= limite:
            ga.reset()
            no_best = 0

    return best
