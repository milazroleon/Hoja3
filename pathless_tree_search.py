class PathlessTreeSearch:
    """
    Implements a pathless tree search that supports BFS and DFS exploration.
    """

    def __init__(self, n0, succ, goal, better=None, order="bfs"):
        self.n0 = n0
        self.succ = succ
        self.goal = goal
        self.better = better
        self.order = order
        self.reset()

    def reset(self):
        self._open = [self.n0]
        self._best = None
        self._active = True

    def step(self):
        if not self._open:
            self._active = False
            return False
        
        if self.order == "bfs":
            node = self._open.pop(0)
        else:
            node = self._open.pop()

        for succ_node in self.succ(node):
            if self.goal(succ_node):
                if self.better is None:
                    self._best = succ_node
                    self._open = []
                    self._active = False
                    return True
                elif self._best is None or self.better(succ_node, self._best):
                    self._best = succ_node
                    return True
            if self.order == "bfs":
                self._open.append(succ_node)
            else:
                self._open.insert(0, succ_node)

        if not self._open:
            self._active = False
        return False

    @property
    def active(self):
        return len(self._open) > 0

    @property
    def best(self):
        return self._best


def encode_problem(domains, constraints, better=None, order="bfs"):
    var_list = list(domains)
    n0 = {}

    def succ(assignment):
        if len(assignment) == len(var_list):
            return []
        next_var = var_list[len(assignment)]
        successors = []
        for val in domains[next_var]:
            new_assign = assignment.copy()
            new_assign[next_var] = val
            if constraints(new_assign):
                successors.append(new_assign)
        return successors

    def goal(assignment):
        return len(assignment) == len(var_list)

    return PathlessTreeSearch(n0=n0, succ=succ, goal=goal, better=better, order=order)