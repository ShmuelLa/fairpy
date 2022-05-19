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
from fairpy.items.allocations_fractional import FractionalAllocation, get_items_of_agent_in_alloc, get_value_of_agent_in_alloc
from networkx.algorithms import bipartite, find_cycle
from networkx.classes.function import create_empty_copy


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
        self.agents = fr_allocation.agents
        self.former_graph = graph
        self.items = items
        self.Gx_complete = None
        self.result_T = None
        self.current_iteration_cycle = None
        self.init_complete_allocation = None


    def find_pareto_improvement(self) -> FractionalAllocation:
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


        TODO Notes:
            call linear prog solve for T each iteration

            use cxvy

            test if a fractional allocation becomes unfrational

            test if a fractional allocation becomes fracitonal up to 1 object
        
        """
        self.__initiate_algorithm_graphs()
        while not self.__is_acyclic():
            for edge in self.current_iteration_cycle:
                tmp_edge = self.linear_programming_solve_optimal_value(edge)
                if tmp_edge is None:
                    continue
                else:
                    self.result_T.add_edge(*tmp_edge)
                    self.Gx_complete.remove_edge(*tmp_edge)
        return self.result_T


    def sum_agents_with_fragment(self, allocation: FractionalAllocation):
        """
        A helper function that sums all the agent with the correspondingfragments allocation
        according to a specific allocation. 
        This function can we run on the former allocation and on the current new complete
        graph allocation which will be removing edges with each linear algorithm iteration.
        """
        result = 0
        for i_agent, agent in enumerate(allocation.agents):
            current_agent_sum = 0
            agent_utilitie_valuation = agent.valuation.map_good_to_value
            agent_fragments = allocation.map_item_to_fraction[i_agent]
            for item in self.items:
                current_agent_sum += agent_fragments[item] * agent_utilitie_valuation[item]
            result += current_agent_sum
        return result

    
    def generate_allocation_from_cycle_edge(self, edge):
        """
        Converts a bipartite graph to a FractionalAllocation object
        This function will be used to prepare the allocation for linear programming
        calculation.

        This function also makes sure that the result is not fragmented 
        as required by the linear solving algporithm conditions
        """
        former_alloc_map = self.former_allocation.map_item_to_fraction
        for agent in self.agents:
            if agent == edge[0]:
                for i_agent, alloc_agent in enumerate(self.agents):
                    if alloc_agent == agent:
                        former_alloc_map[i_agent][edge[1]] = 1.0
                    else:
                        former_alloc_map[i_agent][edge[1]] = 0
        return FractionalAllocation(self.agents, former_alloc_map)


    def __is_acyclic(self) -> bool:
        """
        A helper function which checks if a given bipartite graph has cycles
        it will be used in every iteration in the main algorithms core while loop
        """
        try:
            self.current_iteration_cycle = find_cycle(self.Gx_complete)
            return False
        except nx.NetworkXNoCycle:
            return True


    def __initiate_algorithm_graphs(self):
        """
        A helper function that recevies the current item allocation between agents and 
        creates a complete bipartite graph from them connection all agents with all 
        resources. 
        This function will be used for initializing the main class function.
        """
        result = nx.Graph()
        for agent in self.agents:
            result.add_node(agent)
        for object in self.items:
            result.add_node(object)
        for agent in self.agents:
            for object in self.items:
                result.add_edge(agent, object)
        self.Gx_complete = result
        self.result_T = create_empty_copy(self.Gx_complete, with_data=True)


    def linear_programming_solve_optimal_value(self, edge):
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
        if type(edge[0]) is AdditiveAgent:
            tmp_alloc = self.generate_allocation_from_cycle_edge(edge)
            if self.sum_agents_with_fragment(tmp_alloc) < self.sum_agents_with_fragment(self.former_allocation):
                return None
        return edge


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
