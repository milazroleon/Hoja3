from collections import deque
from copy import deepcopy
from pathless_tree_search import PathlessTreeSearch
import numpy as np


def revise(bcn, X_i, X_j):

    """
    Returns a tuple (D_i', changed), where
        - D_i' is the maximal subset of the domain of X_i that is arc consistent with X_j
        - changed is a boolean value that is True if the domain is now smaller than before and False otherwise

    Args:
        bcn ((domains, constraints)): The BCN containting the constraints, in particular the one for X_i and X_j
        X_i (Any): descriptor of the variable X_i
        X_j (Any): descriptor of the variable X_j
    """

    domains, constraints = bcn
    constraint = constraints.get((X_i, X_j))

    reversed_order = False
    if constraint is None:
        constraint = constraints.get((X_j, X_i))
        reversed_order = True

    old_domain = domains[X_i]
    new_domain = []

    for x in old_domain:
        found = False
        for y in domains[X_j]:
            if reversed_order:
                if constraint(y, x):
                    found = True
                    break
            else:
                if constraint(x, y):
                    found = True
                    break
        if found:
            new_domain.append(x)

    changed = len(new_domain) < len(old_domain)
    domains[X_i] = new_domain
    return new_domain, changed


def ac3(bcn):

    """
    Reduces the domains in a BCN to make it arc consistent, if possible.

    Args:
        bcn ((domains, constraints)): The BCN to make arc consistent (if possible)

    Returns:
        (bcn', feasible), where
        - bcn' is the maximum subnetwork (in terms of domains) of bcn that is consistent
        - feasible is a boolean that is False if it could be verified that bcn doesn't have a solution and True otherwise
    """

    domains, constraints = bcn

    queue = deque()
    for (Xi, Xj) in constraints.keys():
        queue.append((Xi, Xj))
        queue.append((Xj, Xi))

    while queue:
        Xi, Xj = queue.popleft()
        new_domain, changed = revise((domains, constraints), Xi, Xj)

        if not new_domain:
            return (domains, constraints), False

        if changed:
            for (Xk, Xl) in constraints.keys():
                if Xl == Xi and Xk != Xj:
                    queue.append((Xk, Xi))

    return (domains, constraints), True


def get_tree_search_for_bcn(bcn, phi=None):

    """
        Generates a PathlessTreeSearch that can find a solution in the search space described by the BCN.

    Args:
        bcn ((domains, constraints)): The BCN in which we look for a solution.
        phi (func, optional): Function that takes a dictionary of domains (variables are keys) and selects the variable to fix next.

    Returns:
        (search, decoder), where
         - search is a PathlessTreeSearch object
         - decoder is a function to decode a node to an assignment
    """

    from copy import deepcopy

    domains, constraints = bcn

    def goal(current_domains):
        return all(len(v) == 1 for v in current_domains.values())

    def succ(current_domains):
        if phi:
            var = phi(current_domains)
        else:
            var = min(
                (v for v in current_domains if len(current_domains[v]) > 1),
                key=lambda k: len(current_domains[k])
            )

        succesors = []
        for val in current_domains[var]:
            new_dom = deepcopy(current_domains)
            new_dom[var] = [val]
            reducido, posible = ac3((new_dom, constraints))
            if posible:
                succesors.append(reducido[0])
        return succesors

    def decoder(final_domains):
        return {var: vals[0] for var, vals in final_domains.items()}

    return PathlessTreeSearch(
        n0=domains,
        succ=succ,
        goal=goal
    ), decoder

def get_binarized_constraints_for_all_diff(domains):

    """
        Derives all binary constraints that are necessary to make sure that all variables given in `domains` will have different values.

    Args:
        domains (dict): dictionary where keys are variable names and values are lists of possible values for the respective variable.

    Returns:
        dict: dictionary where keys are constraint names (it is recommended to use tuples, with entries in the tuple being the variable names sorted lexicographically) and values are the functions encoding the respective constraint set membership
    """

    constraints = {}
    variables = sorted(domains.keys())
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            Xi, Xj = variables[i], variables[j]
            def make_neq():
                def neq(a, b):
                    return a != b
                return neq
            constraints[(Xi, Xj)] = make_neq()

    return constraints
