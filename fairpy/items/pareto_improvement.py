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

from fairpy.agents import AdditiveAgent
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

    HW4 Tests:

    Example 1:
    >>> agent1 = AdditiveAgent({"a": 2, "b": 4, "c": 3}, name="agent1")
    >>> list_of_agents_for_func = [agent1]
    >>> items_for_func ={'a','b','c'}
    >>> alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
    >>> fpo_alloc = find_pareto_improvement(alloc_y_for_func, items_for_func)
    >>> print(fpo_alloc)
    agent1's bundle: {a,b,c},  value: 9.0


    >>> agent1 = AdditiveAgent({"a": -2, "b": -4, "c": -3}, name="agent1")
    >>> list_of_agents_for_func = [agent1]
    >>> items_for_func ={'a','b','c'}
    >>> alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
    >>> fpo_alloc = find_pareto_improvement(alloc_y_for_func, items_for_func)
    >>> print(fpo_alloc)
    agent1's bundle: {a,b,c},  value: -9.0


    Example 2:
    >>> agent1 = AdditiveAgent({"a": 1, "b1": 1, "b2": 1}, name="agent1")
    >>> agent2 = AdditiveAgent({"a": 1, "b1": 1, "b2": 1}, name="agent2")
    >>> all_agents = [agent1, agent2]
    >>> all_items = {'a', 'b1', 'b2'}
    >>> alloc_y_for_func = FractionalAllocation(all_agents, [{"a": 0.5, "b1": 0.25, "b2": 0.25},{"a": 0.4, "b1": 0.3, "b2": 0.3}])
    >>> fpo_alloc = find_pareto_improvement(alloc_y_for_func)
    >>> print(alloc)
    agent1's bundle: {a},  value: 0.5
    agent2's bundle: {b},  value: 0.6


    Example 3:
    >>> agent1 = AdditiveAgent({"a": 1, "b": 1, "c": 1.3}, name="agent1")
    >>> agent2 = AdditiveAgent({"a": 1, "b": 1, "c": 1.3}, name="agent2")
    >>> agent3 = AdditiveAgent({"a": 1, "b": 1, "c": 1.3}, name="agent3")
    >>> all_agents = [agent1, agent2, agent3]
    >>> all_items = {'a', 'b', 'c'}
    >>> alloc_y_for_func = FractionalAllocation(all_agents, [{"a": 0.2, "b": 0.3, "c": 0.5},{"a": 0.4, "b": 0.1, "c": 0.5},{"a": 0.4, "b": 0.4, "c": 0.5}])
    >>> fpo_alloc = find_pareto_improvement(alloc_y_for_func)
    >>> print(alloc)
    agent1's bundle: {c},  value: 0.5
    agent2's bundle: {a},  value: 0.4
    agent3's bundle: {b},  value: 0.4
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

if __name__ == "__main__":
    import doctest
    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))