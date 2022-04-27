#!python3

"""
An implementation of a Pareto Improvement allocation algorithm.
This algorithm is a key improvement feature in the implemented article.

Reference:

    Haris Aziz, Herve Moulin and Fedor Sandomirskiy (2020).
    ["A polynomial-time algorithm for computing a Pareto optimal and almost proportional allocation"](https://arxiv.org/pdf/1909.00740.pdf).
    Operations Research Letters.
    * Algorithm 2, Step 2 in Algorithm 1

Programmer: Shmuel Lavian
Since:  2022-04
"""

from fairpy.items.allocations_fractional import FractionalAllocation
from networkx.algorithms import bipartite


def find_pareto_improvement(fpo_alloc: FractionalAllocation) -> FractionalAllocation:
    """
    This is the modules main function which implements the 2nd algorithm presented in the article and is used as the
    second stage in the po_and_prop1_allocation algorithm.

    INPUT:
    * fpo_alloc: A Pareto-optimal (PO) allocation corresponding to the given consumption graph.
    The allocation is an allocation instance that includes sets of:
        (Agents, Objects and their corresponding allocation)

    OUTPUT:
    * Fractional-Pareto-Optimal (fPO) that improves the former given allocation instance for which
    the allocation graph Gx is acyclic
    """

def is_acyclic(Gx: bipartite) -> bool:
    """
    A help function wchich checks if a given bipartite graph has cycles
    it will be used in every iteration in the main algorithms core while loop
    """

def linear_programming_solve_optimal_value(fpo_alloc: FractionalAllocation) -> tuple:
    """
    Main linear programming help function which will receive the current allocation
    and find an optimal set for the current states results according to four mathematical
    conditions represented in Algorithms 2.

    OUTPUT:
    * A tuple containing (bool, optimal_value).
        The boolean value will be used as a flag in order to mark it the optimal value was found
        if true, the value was found and will be placed in the tuple's second index.
        else, the boolean flag will be false and the second index will be empty.
        this will point the main function to stop.
    """