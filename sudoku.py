from collections import defaultdict

from ac import get_binarized_constraints_for_all_diff

def get_bcn_for_sudoku(sudoku):

    """
        Receives a Sudoku and creates a BCN definition from it, using the binarized All-Diff Constraint

    Args:
        sudoku (np.ndarray): numpy array describing the sudoku

    Returns:
        (domains, constraints): BCN describing the conditions for the given Sudoku
    """

    n = sudoku.shape[0]                
    box_size = int(n ** 0.5)           

    domains = {}
    constraints = {}

    for r in range(n):
        for c in range(n):
            var = f"R{r}C{c}"
            if sudoku[r, c] != 0:
                domains[var] = [sudoku[r, c]]
            else:
                domains[var] = list(range(1, n + 1))

    for r in range(n):
        fila_vars = {f"R{r}C{c}": domains[f"R{r}C{c}"] for c in range(n)}
        constraints.update(get_binarized_constraints_for_all_diff(fila_vars))

    for c in range(n):
        col_vars = {f"R{r}C{c}": domains[f"R{r}C{c}"] for r in range(n)}
        constraints.update(get_binarized_constraints_for_all_diff(col_vars))

    for br in range(0, n, box_size):
        for bc in range(0, n, box_size):
            block_vars = {}
            for r in range(br, br + box_size):
                for c in range(bc, bc + box_size):
                    var = f"R{r}C{c}"
                    block_vars[var] = domains[var]
            constraints.update(get_binarized_constraints_for_all_diff(block_vars))

    return domains, constraints