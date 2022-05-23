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

from re import A
from unittest import result
import cvxpy
from fairpy import ValuationMatrix
import matplotlib.pyplot as plt
import networkx as nx
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
        self.former_allocation = fr_allocation
        self.agents = fr_allocation.agents
        self.items = items
        self.Gx_complete = None
        self.result_T = None
        self.current_iteration_cycle = None
        self.val_mat = convert_FractionalAllocation_to_ValuationMatrix(self.former_allocation)


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

        Example 1 (One agent - positive/negative/mixed object allocation, will get everything):
        >>> agent1 = AdditiveAgent({"x": 1, "y": 2, "z": 4}, name="agent1")
        >>> agents = [agent1]
        >>> items_for_func ={'x','y','z'}
        >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':1.0, 'z':1.0}])
        >>> G = nx.Graph()
        >>> G.add_node(agent1)
        >>> G.add_node('x')
        >>> G.add_node('y')
        >>> G.add_node('z')
        >>> G.add_edge(agent1, 'x')
        >>> G.add_edge(agent1, 'y')
        >>> G.add_edge(agent1, 'z')
        >>> pi = ParetoImprovement(allocations, G, items_for_func)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True

        >>> agent1 = AdditiveAgent({"x": -1, "y": -2, "z": -4}, name="agent1")
        >>> agents = [agent1]
        >>> items_for_func ={'x','y','z'}
        >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':1.0, 'z':1.0}])
        >>> G = nx.Graph()
        >>> G.add_node(agent1)
        >>> G.add_node('x')
        >>> G.add_node('y')
        >>> G.add_node('z')
        >>> G.add_edge(agent1, 'x')
        >>> G.add_edge(agent1, 'y')
        >>> G.add_edge(agent1, 'z')
        >>> pi = ParetoImprovement(allocations, G, items_for_func)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True

        >>> agent1 = AdditiveAgent({"x": -1, "y": 2, "z": -4}, name="agent1")
        >>> agents = [agent1]
        >>> items_for_func ={'x','y','z'}
        >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':1.0, 'z':1.0}])
        >>> G = nx.Graph()
        >>> G.add_node(agent1)
        >>> G.add_node('x')
        >>> G.add_node('y')
        >>> G.add_node('z')
        >>> G.add_edge(agent1, 'x')
        >>> G.add_edge(agent1, 'y')
        >>> G.add_edge(agent1, 'z')
        >>> pi = ParetoImprovement(allocations, G, items_for_func)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True

        Example 2 (3rd example from the article)
        >>> agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
        >>> agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
        >>> G = nx.Graph()
        >>> agents = [agent1, agent2]
        >>> items = {'a', 'b', 'c', 'd'}
        >>> G.add_nodes_from(agents + list(items))
        >>> G.add_edges_from([[agent1, 'b'], [agent1, 'c'], [agent2, 'a'], [agent2, 'b'], [agent2, 'd']])
        >>> allocations = FractionalAllocation(agents, [{'a':0.0,'b':0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
        >>> pi = ParetoImprovement(allocations, G, items)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True
        
        Main algorithm 1 developper test
        >>> agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
        >>> agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
        >>> all_items  = {'a', 'b', 'c', 'd'}
        >>> all_agents = [agent1, agent2]
        >>> initial_allocation = FractionalAllocation(all_agents, [{'a':0.0,'b': 0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
        >>> G = nx.Graph()
        >>> G.add_node(agent1)
        >>> G.add_node(agent2)
        >>> G.add_node('a')
        >>> G.add_node('b')
        >>> G.add_node('c')
        >>> G.add_node('d')
        >>> G.add_edge(agent1, 'b')
        >>> G.add_edge(agent1, 'c')
        >>> G.add_edge(agent2, 'a')
        >>> G.add_edge(agent2, 'b')
        >>> G.add_edge(agent2, 'd')
        >>> pi = ParetoImprovement(initial_allocation, G, all_items)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True
        """
        if len(self.agents) == 1:
            return self.former_allocation
        self.__initiate_algorithm_graphs()
        while not self.__is_acyclic():
            for edge in self.current_iteration_cycle:
                tmp_edge, tmp_iptimum = self.__linear_prog_solve(edge)
                if tmp_edge is None or tmp_iptimum is None:
                    continue
                else:
                    tmp_result = self.result_T
                    tmp_result.add_edge(*tmp_edge)
                    if is_acyclic(self.result_T) and is_acyclic(tmp_result):
                        self.result_T.add_edge(*tmp_edge)
                        self.Gx_complete.remove_edge(*tmp_edge)
        # plot_graph(self.result_T)
        return self.__convert_result_graph_to_FractionalAllocation()


    def __linear_prog_solve(self, edge):
        """
        Main linear programming help function which will receive the current allocation
        and find an optimal set for the current states results according to four mathematical
        conditions represented in Algorithms 2.

        OUTPUT:
        * A tuple containing (edge, optimal_value).
            The edge will be used to create and test the result graph for cycles with or without it
        """
        if type(edge[0]) is AdditiveAgent:
            tmp_alloc = self.__generate_allocation_from_cycle_edge(edge)
            tmp_mat = convert_FractionalAllocation_to_ValuationMatrix(tmp_alloc)

            # Creates [agens, object] constrained matrix for variable input
            allocation_vars = cvxpy.Variable((tmp_mat.num_of_agents, tmp_mat.num_of_objects))

            # line 5 in the algorithm, the max condition
            sum_x = sum(allocation_vars[i][o] * tmp_mat[i][o] for o in tmp_mat.objects() for i in tmp_mat.agents())
            sum_y = sum(allocation_vars[i][o] * self.val_mat[i][o] for o in self.val_mat.objects() for i in self.val_mat.agents())

            first_constraints = [sum_x >= sum_y]
            positivity_constraints = [
                allocation_vars[i][o] >= 0 for i in tmp_mat.agents()
                for o in tmp_mat.objects()
            ]
            feasibility_constraints = [
                sum([allocation_vars[i][o] for i in tmp_mat.agents()])==1
                for o in tmp_mat.objects()
            ]
            constraints = first_constraints + positivity_constraints + feasibility_constraints
            problem = cvxpy.Problem(cvxpy.Maximize(sum_x), constraints)
            opt = problem.solve(
                #Uncomment next line in order to see all solving comments
                # verbose=True
                )
            return edge, opt
        return None, None

    
    def __generate_allocation_from_cycle_edge(self, edge):
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
        this class method is especialy used in order to save each iteration cycle
        for edge testing in the algorithm
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
        Gx_complete = nx.Graph()
        for agent in self.agents:
            Gx_complete.add_node(agent)
        for object in self.items:
            Gx_complete.add_node(object)
        for agent in self.agents:
            for object in self.items:
                Gx_complete.add_edge(agent, object)
        self.Gx_complete = Gx_complete
        self.result_T = create_empty_copy(self.Gx_complete, with_data=True)


    def __convert_result_graph_to_FractionalAllocation(self) -> FractionalAllocation:
        """
        Converts the resulting allocation graph T ro a FractionalAllocation
        object for the main articles algorithm to work on
        """
        result_allocation_list = []
        for agent in self.agents:
            tmp_agent_allocation_map = {}
            for item in self.items:
                tmp_agent_allocation_map[item] = 0
            for item in self.result_T.neighbors(agent):
                tmp_agent_allocation_map[item] = 1
            result_allocation_list.append(tmp_agent_allocation_map)
        return FractionalAllocation(self.agents, result_allocation_list)


def plot_graph(graph):
    """
    Draws the received networkx graph, this functon is used for visual 
    testing during development
    """
    nx.draw(graph, with_labels = True)
    plt.show()


def is_acyclic(graph) -> bool:
    """
    A helper function which checks if a given graph has cycles
    this function will be used to check the result graph for cycles
    """
    try:
        find_cycle(graph)
        return False
    except nx.NetworkXNoCycle:
        return True


def convert_FractionalAllocation_to_ValuationMatrix(allocation: FractionalAllocation) -> ValuationMatrix:
    """
    A helper function that converts a FractionalAllocation object to ValuationMatrix
    This function is used to prepare the matrix parameters for the linear solving process 
    """
    matrix_allocation_list = []
    for agent_alloc in allocation.map_item_to_fraction:
        agent_alloc_list = []
        for _, alloc in agent_alloc.items():
            agent_alloc_list.append(alloc)
        matrix_allocation_list.append(agent_alloc_list)
    v_mat = ValuationMatrix(matrix_allocation_list)
    return v_mat


if __name__ == "__main__":
    import doctest
    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))