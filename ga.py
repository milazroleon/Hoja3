from functools import cmp_to_key


class GeneticSearch:
    """
        Optimizes a set of candidates explorable through crossovers and mutations.

        Which candidate is better is determined with the `better` function
    """

    def __init__(self, init, crossover, mutate, better, population_size):
        self.init = init
        self.crossover = crossover
        self.mutate = mutate
        self.better = better
        self.population_size = population_size

        # state variables
        self.population = None
        self._best = None
        self.num_solutions = 0

    def reset(self):
        self.population = list(self.init(self.population_size))
        self.num_solutions = len(self.population)
        if self.population:
            best = self.population[0]
            for ind in self.population[1:]:
                if self.better(ind, best):
                    best = ind
            self._best = best
        else:
            self._best = None

    @property
    def best(self):
        return self._best

    @property
    def active(self):
        return self.population is not None

    def step(self):
        if not self.active:
            raise RuntimeError("Population not initialized. Call reset() first.")

        parents = self.population
        children = []
        n = len(parents)
        for i in range(n):
            for j in range(i + 1, n):
                    off = self.crossover(parents[i], parents[j])
                    for child in off:
                        if self.mutate is not None:
                            child = self.mutate(child)
                        children.append(child)

        combined = parents + children

        combined_sorted = sorted(combined, key=cmp_to_key(self._cmp))
        new_pop = combined_sorted[: self.population_size]

        self.population = new_pop
        self.num_solutions += len(children)

        best = self._best
        for ind in self.population:
            if best is None or self.better(ind, best):
                best = ind
        self._best = best


    def _cmp(self, a, b):
        if self.better(a, b):
            return -1
        if self.better(b, a):
            return 1
        return 0
