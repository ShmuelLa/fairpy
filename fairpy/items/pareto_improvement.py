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

from unittest import result
import networkx as nx
import matplotlib.pyplot as plt
from fairpy.agents import AdditiveAgent, Bundle
from fairpy.items.allocations_fractional import FractionalAllocation
from networkx.algorithms import bipartite, find_cycle, complete_bipartite_graph


class ParetoImprovement:
    """
    Main pareto Improvement class, will nest methods in order to calculate pareto improvement.
    """

    def __init__(self, fr_allocation: FractionalAllocation, graph: bipartite, items: Bundle):
        """
        These are the main loops and values for linear programming inputs 
        that are stored for later calculation and testing.
        """
        self.linear_calc_objects = 0.0
        self.linear_agents_fragments = {}
        self.linear_agent_fragments = {}
        self.former_allocation = fr_allocation
        self.former_graph = graph
        self.items = items
        self.Gx_complete = None
        self.result_T = nx.Graph()


    def find_pareto_improvement(self) -> FractionalAllocation:
        self.__construct_complete_bipartite_graph()
        while not self.__is_acyclic():
            print("zbi")
        print(result_T)
        nx.draw(result_T, with_labels = True)
        plt.show()
        return self.result


    def __is_acyclic(self) -> bool:
        """
        A helper function which checks if a given bipartite graph has cycles
        it will be used in every iteration in the main algorithms core while loop

            NOTE:This si the only function that won't be tested at this stage because
            it's based on networkx.
        """
        try:
            find_cycle(self.Gx_complete)
            return False
        except nx.NetworkXNoCycle:
            return True


    def __construct_complete_bipartite_graph(self):
        """
        A helper function that recevies the current item allocation between agents and 
        creates a complete bipartite graph from them connection all agents with all 
        resources. 
        This function will be used for initializing the main class function.
        """
        result = nx.Graph()
        for agent in self.former_allocation.agents:
            result.add_node(agent)
        for object in self.items:
            result.add_node(object)
        for agent in self.former_allocation.agents:
            for object in self.items:
                result.add_edge(agent, object)
        self.Gx_complete = result



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

    agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
    agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
    all_items  = {'a', 'b', 'c', 'd'}
    all_agents = [agent1, agent2]
    initial_allocation = FractionalAllocation(all_agents, 
    [{'a':0.0,'b': 0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
    G = nx.Graph()
    G.add_node(agent1)
    G.add_node(agent2)
    G.add_node('a')
    G.add_node('b')
    G.add_node('c')
    G.add_node('d')
    G.add_edge(agent1, 'b')
    G.add_edge(agent1, 'c')
    G.add_edge(agent2, 'a')
    G.add_edge(agent2, 'b')
    G.add_edge(agent2, 'd')

    pi = ParetoImprovement(initial_allocation, G, all_items)
    pi.find_pareto_improvement()
    

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
>>> f_alloc = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
>>> fpo_alloc = find_pareto_improvement(f_alloc)
>>> print(fpo_alloc)
agent1's bundle: {a,b,c},  value: 9.0


>>> agent1 = AdditiveAgent({"a": -2, "b": -4, "c": -3}, name="agent1")
>>> list_of_agents_for_func = [agent1]
>>> items_for_func ={'a','b','c'}
>>> f_alloc = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
>>> fpo_alloc = find_pareto_improvement(f_alloc)
>>> print(fpo_alloc)
agent1's bundle: {a,b,c},  value: -9.0


Example 2:
>>> agent1 = AdditiveAgent({"a": 1, "b1": 1, "b2": 1}, name="agent1")
>>> agent2 = AdditiveAgent({"a": 1, "b1": 1, "b2": 1}, name="agent2")
>>> all_agents = [agent1, agent2]
>>> all_items = {'a', 'b1', 'b2'}
>>> f_alloc = FractionalAllocation(all_agents, [{"a": 0.5, "b1": 0.25, "b2": 0.25},{"a": 0.4, "b1": 0.3, "b2": 0.3}])
>>> fpo_alloc = find_pareto_improvement(f_alloc)
>>> print(fpo_alloc)
agent1's bundle: {a},  value: 0.5
agent2's bundle: {b},  value: 0.6


Example 3:
>>> agent1 = AdditiveAgent({"a": 1, "b": 1, "c": 1.3}, name="agent1")
>>> agent2 = AdditiveAgent({"a": 1, "b": 1, "c": 1.3}, name="agent2")
>>> agent3 = AdditiveAgent({"a": 1, "b": 1, "c": 1.3}, name="agent3")
>>> all_agents = [agent1, agent2, agent3]
>>> all_items = {'a', 'b', 'c'}
>>> f_alloc = FractionalAllocation(all_agents, [{"a": 0.2, "b": 0.3, "c": 0.5},{"a": 0.4, "b": 0.1, "c": 0.5},{"a": 0.4, "b": 0.4, "c": 0.5}])
>>> fpo_alloc = find_pareto_improvement(f_alloc)
>>> print(fpo_alloc)
agent1's bundle: {c},  value: 0.5
agent2's bundle: {a},  value: 0.4
agent3's bundle: {b},  value: 0.4
"""
