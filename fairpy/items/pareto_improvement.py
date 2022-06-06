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

import cvxpy
import matplotlib.pyplot as plt
import networkx as nx
from numpy import indices

from fairpy import ValuationMatrix
from allocation_matrix import AllocationMatrix
from fairpy.agents import AdditiveAgent, Bundle
from fairpy.items.allocations_fractional import FractionalAllocation
from networkx.algorithms import find_cycle, bipartite


class ParetoImprovement:
    """
    Main pareto Improvement class, will nest methods in order to calculate pareto improvement.
    This class will be called from the find_po_and_prop1_allocation function as the 2nd 
    algorithmic step for improving the allocation
    """
    def __init__(self, fr_allocation: FractionalAllocation, items: Bundle):
        """
        These are the main loops and values for linear programming inputs 
        that are stored for later calculation and testing.
        """
        self.former_allocation = fr_allocation
        self.agents = fr_allocation.agents
        self.items = items
        self.Gx = None
        self.result_T = set()
        self.x_allocations = {}
        self.current_iteration_cycle = None
        self.val_mat = convert_FractionalAllocation_to_ValuationMatrix(self.former_allocation)
        self.agent_indices = {}
        self.item_indices = {}


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
        >>> pi = ParetoImprovement(allocations, items_for_func)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True

        >>> agent1 = AdditiveAgent({"x": -1, "y": -2, "z": -4}, name="agent1")
        >>> agents = [agent1]
        >>> items_for_func ={'x','y','z'}
        >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':1.0, 'z':1.0}])
        >>> pi = ParetoImprovement(allocations, items_for_func)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True

        >>> agent1 = AdditiveAgent({"x": -1, "y": 2, "z": -4}, name="agent1")
        >>> agents = [agent1]
        >>> items_for_func ={'x','y','z'}
        >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':1.0, 'z':1.0}])
        >>> pi = ParetoImprovement(allocations, items_for_func)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True

        Example 2 (3rd example from the article)
        >>> agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
        >>> agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
        >>> agents = [agent1, agent2]
        >>> items = {'a', 'b', 'c', 'd'}
        >>> allocations = FractionalAllocation(agents, [{'a':0.0,'b':0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
        >>> pi = ParetoImprovement(allocations, items)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True
        
        Main algorithm 1 developer test
        >>> agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
        >>> agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
        >>> all_items  = {'a', 'b', 'c', 'd'}
        >>> all_agents = [agent1, agent2]
        >>> initial_allocation = FractionalAllocation(all_agents, [{'a':0.0,'b': 0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
        >>> pi = ParetoImprovement(initial_allocation, all_items)
        >>> pi.find_pareto_improvement().is_complete_allocation()
        True
        """
        if len(self.agents) == 1:
            return self.former_allocation
        self.Gx, self.x_allocations, self.agent_indices, self.item_indices = convert_FractionalAllocation_to_graph(self.former_allocation, True)
        while not self.__is_acyclic():
            for edge in self.current_iteration_cycle:
                tmp_opt = self.__linear_prog_solve(self.result_T)
                tmp_T = self.result_T
                # tmp_T = add_edge_to_set(edge, tmp_T, self.Gx_complete.get_edge_data(edge[0], edge[1])['weight'])
                tmp_T.add(edge)
                tmp_opt2 = self.__linear_prog_solve(tmp_T)
                if tmp_opt == tmp_opt2:
                    # add_edge_to_set(edge, self.result_T, self.Gx_complete.get_edge_data(edge[0], edge[1])['weight'])
                    self.result_T.add(edge)
            self.Gx, self.x_allocations, self.agent_indices, self.item_indices = convert_FractionalAllocation_to_graph(
                allocation=convert_set_to_FractionalAllocation(self.agents, self.x_allocations, self.result_T),
                complete_flag=False)
            # print(self.result_T)
        return convert_set_to_FractionalAllocation(self.agents, self.x_allocations, self.result_T)

    # def find_pareto_improvement2(self) -> FractionalAllocation:
    #     # V2 With cycle edge removals:
    #     if len(self.agents) == 1:
    #         return self.former_allocation
    #     self.Gx, self.x_allocations = convert_FractionalAllocation_to_graph(self.former_allocation, True)
    #     while not self.__is_acyclic():
    #         for edge in self.current_iteration_cycle:
    #             tmp_opt = self.__linear_prog_solve(self.result_T)
    #             tmp_T = self.result_T
    #             # tmp_T = add_edge_to_set(edge, tmp_T, self.Gx_complete.get_edge_data(edge[0], edge[1])['weight'])
    #             tmp_T.add(edge)
    #             tmp_opt2 = self.__linear_prog_solve(tmp_T)
    #             if tmp_opt == tmp_opt2:
    #                 # add_edge_to_set(edge, self.result_T, self.Gx_complete.get_edge_data(edge[0], edge[1])['weight'])
    #                 self.result_T.add(edge)
    #                 self.Gx.remove_edge(edge[0], edge[1])
    #         _, self.x_allocations = convert_FractionalAllocation_to_graph(
    #             allocation=convert_set_to_FractionalAllocation(self.agents, self.x_allocations, self.result_T),
    #             complete_flag=False)
    #         # print(self.result_T)
    #     return convert_set_to_FractionalAllocation(self.agents, self.x_allocations, self.result_T)


    def __linear_prog_solve(self, result_set: set) -> float:
        """
        Main linear programming help function which will receive the current allocation
        and find an optimal set for the current states results according to four mathematical
        conditions represented in Algorithms 2.

        OUTPUT:
        * A tuple containing (edge, optimal_value).
            The edge will be used to create and test the result graph for cycles with or without it
        """
        y_former_valuation_mat = convert_FractionalAllocation_to_ValuationMatrix(self.former_allocation)
        y_former_allocation_mat = AllocationMatrix(self.former_allocation)

        # if len(result_set) != 0:
        #     x_valuation_mat = convert_FractionalAllocation_to_ValuationMatrix(convert_set_to_FractionalAllocation(self.agents ,self.x_allocations ,result_set))
        #     x_allocation_matrix = AllocationMatrix(convert_set_to_FractionalAllocation(self.agents ,self.x_allocations ,result_set))
        # else:
        #     x_valuation_mat = convert_FractionalAllocation_to_ValuationMatrix(self.former_allocation)
        #     x_allocation_matrix = AllocationMatrix(self.former_allocation)

        
        # Creates [agents, object] constrained matrix for variable input

        x_allocation_vars = cvxpy.Variable((self.val_mat.num_of_agents, self.val_mat.num_of_objects))


        sum_x = [sum(x_allocation_vars[i][o] * self.val_mat[i][o] for o in self.val_mat.objects()) for i in self.val_mat.agents()]
        sum_y = [sum(y_former_allocation_mat[i][o] * self.val_mat[i][o] for o in self.val_mat.objects()) for i in self.val_mat.agents()]

        first_constraints = [sum_x[i] >= sum_y[i] for i in self.val_mat.agents()]

        feasibility_constraints = [
            sum([x_allocation_vars[i][o] for i in self.val_mat.agents()]) == 1
            for o in self.val_mat.objects()
        ]

        positivity_constraints = [
            x_allocation_vars[i][o] >= 0 
            for i in self.val_mat.agents()
            for o in self.val_mat.objects()
        ]

        zero_constraints = []
        for edge in result_set:
            i, o = None, None
            if type(edge[0]) == AdditiveAgent:
                i = self.agent_indices[edge[0]]
                o = self.item_indices[edge[1]]
            else:
                i = self.agent_indices[edge[1]]
                o = self.item_indices[edge[0]]
            zero_constraints.append(x_allocation_vars[i][o]==0)

        constraints = first_constraints + positivity_constraints + feasibility_constraints + zero_constraints + [sum(sum_x)<=1000]

        problem = cvxpy.Problem(cvxpy.Maximize(sum(sum_x)), constraints)
        problem.solve(
            #Uncomment next line in order to see all solving comments
            # verbose=True
            )
        if not problem.status=="optimal": 
            raise ValueError(f"Problem status is {problem.status}")
        # x_allocation_values = [[x_allocation_vars[i][o].value for o in x_valuation_mat.objects()] for i in x_valuation_mat.agents()]
        return problem.value

    
    # def __linear_prog_solve2(self, result_set: set) -> float:
    #     result = float()
    #     results = set()

    #     y_former_valuation_mat = convert_FractionalAllocation_to_ValuationMatrix(self.former_allocation)
    #     y_former_allocation_mat = AllocationMatrix(self.former_allocation)

    #     if len(result_set) != 0:
    #         x_valuation_mat = convert_FractionalAllocation_to_ValuationMatrix(convert_set_to_FractionalAllocation(self.agents ,self.x_allocations ,result_set))
    #         x_allocation_matrix = AllocationMatrix(convert_set_to_FractionalAllocation(self.agents ,self.x_allocations ,result_set))
    #     else:
    #         x_valuation_mat = convert_FractionalAllocation_to_ValuationMatrix(self.former_allocation)
    #         x_allocation_matrix = AllocationMatrix(self.former_allocation)

    #     sum_x = sum(x_allocation_matrix[i][o] * x_valuation_mat[i][o] 
    #         for o in x_valuation_mat.objects() 
    #         for i in x_valuation_mat.agents())

    #     sum_y = sum(y_former_allocation_mat[i][o] * self.val_mat[i][o] 
    #         for o in self.val_mat.objects() 
    #         for i in self.val_mat.agents())
        
    #     allocation_sum = sum(x_allocation_matrix[i][o] 
    #         for i in x_valuation_mat.agents()
    #         for o in x_valuation_mat.objects())

    #     for i in x_valuation_mat.num_of_agents:
    #         for o in x_valuation_mat.num_of_objects:
    #             if sum_x >= sum_y and allocation_sum == 1:
    #                 if x_allocation_matrix[i, o] == 0:
    #                     return


    def __is_acyclic(self) -> bool:
        """
        A helper function which checks if a given bipartite graph has cycles
        it will be used in every iteration in the main algorithms core while loop
        this class method is especially used in order to save each iteration cycle
        for edge testing in the algorithm

        Returns:
            bool: True if self Gx graph is acyclic False otherwise
        """
        try:
            self.current_iteration_cycle = find_cycle(self.Gx)
            return False
        except nx.NetworkXNoCycle:
            return True


def get_x_allocation_value(x_allocations: dict, edge: tuple) -> float:
    """
    A helper function that's used to receive a specific agent -> item
    allocation value from 0.0 to 1.0. This function is needed because of
    networkx inconsistency with edge insertion

    Args:
        x_allocations (dict): the x* allocation dictionary containing 
            ((agent, item) or (item, agent)) keys and allocation values as values 
        edge (tuple): a tuple of (agent, item) or (item, agent)

    Returns:
        float: The specific agent to item allocation value

    
    >>> agent1 = AdditiveAgent({'x':1, 'y':2, 'z':3}, name="agent1")
    >>> agent2 = AdditiveAgent({'x':3, 'y':2, 'z':1}, name="agent2")
    >>> x_alloc = {(agent1, 'x'): 0.3, ('x', agent2): 0.7}
    >>> get_x_allocation_value(x_alloc ,(agent1, 'x'))
    0.3
    >>> get_x_allocation_value(x_alloc ,(agent2, 'x'))
    0.7
    """
    try: 
        return x_allocations[edge]
    except KeyError:
        try: 
            return x_allocations[(edge[1], edge[0])]
        except KeyError:
            return 0
        

def convert_FractionalAllocation_to_graph(allocation: FractionalAllocation, complete_flag: bool) -> nx.Graph:
    """
    A helper function that converts a FractionalAllocation object to NxGraph while
    saving all the current allocation values to a dict and delivering them to.
    
    :param complete_flag: when true, will create a complete bipartite graph
    with the same agents and items. 
    otherwise will create the graph with connections where the allocation value is 
    higher than zero


    >>> agent1 = AdditiveAgent({'x':1, 'y':2, 'z':3}, name="agent1")
    >>> agent2 = AdditiveAgent({'x':3, 'y':2, 'z':1}, name="agent2")
    >>> allocation = FractionalAllocation([agent1, agent2], [{'x':0.5, 'y':0.5, 'z':0.5},{'x':0.5, 'y':0.5, 'z':0.5}])
    >>> gx, x_alloc, agents_i, items_i  = convert_FractionalAllocation_to_graph(allocation, True)
    >>> bipartite.is_bipartite(gx)
    True
    >>> for k, v in x_alloc.items():
    ...     print(k, v)
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'x') 0.5
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'y') 0.5
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'z') 0.5
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'x') 0.5
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'y') 0.5
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'z') 0.5
    >>> agents_i[agent1]
    0
    >>> agents_i[agent2]
    1
    >>> items_i['x']
    0
    >>> items_i['y']
    1
    >>> items_i['z']
    2
    >>> allocation = FractionalAllocation([agent1, agent2], [{'x':0.5, 'y':1, 'z':0.5},{'x':0.5, 'y':0, 'z':0.5}])
    >>> gx, x_alloc, agents_i, items_i  = convert_FractionalAllocation_to_graph(allocation, False)
    >>> bipartite.is_bipartite(gx)
    True
    >>> for k, v in x_alloc.items():
    ...     print(k, v)
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'x') 0.5
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'y') 1
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'z') 0.5
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'x') 0.5
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'y') 0
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'z') 0.5
    >>> for edge in gx.edges():
    ...     print(edge)
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'x')
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'y')
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'z')
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'x')
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'z')
    >>> allocation = FractionalAllocation([agent1, agent2], [{'x':0, 'y':1, 'z':0},{'x':1, 'y':0, 'z':1}])
    >>> gx, x_alloc, agents_i, items_i  = convert_FractionalAllocation_to_graph(allocation, False)
    >>> bipartite.is_bipartite(gx)
    True
    >>> for edge in gx.edges():
    ...     print(edge)
    (agent1 is an agent with a Additive valuation: x=1 y=2 z=3., 'y')
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'x')
    (agent2 is an agent with a Additive valuation: x=3 y=2 z=1., 'z')
    """
    Gx = nx.Graph()
    x_allocations = {}
    agent_indices = {}
    item_indices = {}
    items = allocation.map_item_to_fraction[0].keys()
    for i ,agent in enumerate(allocation.agents):
        Gx.add_node(agent)
        agent_indices[agent] = i
    for i, object in enumerate(items):
        Gx.add_node(object)
        item_indices[object] = i
    for agent in allocation.agents:
        for object in items:
            x_allocations[(agent, object)] = get_agent_fraction(allocation, agent, object)
            if complete_flag or get_x_allocation_value(x_allocations, (agent, object)) > 0:
                Gx.add_edge(agent, object)
    return Gx, x_allocations, agent_indices, item_indices

    
def convert_set_to_FractionalAllocation(agents: list, x_allocations: dict, allocation_set: set) -> FractionalAllocation:
    """
    Converts an allocation set to a FractionalAllocation object
    this helper function is mainly used to convert the result_T allocation set to 
    and allocation object for the algorithm cycles


    >>> agent1 = AdditiveAgent({'x':1, 'y':2, 'z':3}, name="agent1")
    >>> agent2 = AdditiveAgent({'x':3, 'y':2, 'z':1}, name="agent2")
    >>> test_set = set()
    >>> test_set.add((agent1, 'x'))
    >>> test_set.add((agent2, 'y'))
    >>> test_set.add((agent2, 'z'))
    >>> x_allocations = {(agent1, 'x'): 1, (agent2, 'y'): 1, (agent2, 'z'): 1}
    >>> agents = [agent1, agent2]
    >>> allocation = convert_set_to_FractionalAllocation(agents, x_allocations, test_set)
    >>> allocation
    agent1's bundle: {x: 1, y: 0, z: 0},  value: 1
    agent2's bundle: {x: 0, y: 1, z: 1},  value: 3
    <BLANKLINE>
    """
    allocation_list = []
    for agent in agents:
        current_agent_alloc_dict = create_empty_item_allocation_dict(agents[0].valuation.map_good_to_value.keys())
        for alloc_edge in allocation_set:
            if agent == alloc_edge[0]:
                current_agent_alloc_dict[alloc_edge[1]] = get_x_allocation_value(x_allocations, alloc_edge)
            elif agent == alloc_edge[1]:
                current_agent_alloc_dict[alloc_edge[0]] = get_x_allocation_value(x_allocations, alloc_edge)
        allocation_list.append(current_agent_alloc_dict)
    return FractionalAllocation(agents, allocation_list)


def create_empty_item_allocation_dict(items: set) -> dict:
    """
    A helper function that creates an item allocation dictionary
    this function is used in order to create agent allocation dictionaries
    for FractionalAllocation building
    """
    result = {}
    for item in items:
        result[item] = 0
    return result


def plot_graph(graph):
    """
    Draws the received networkx graph, this function is used for visual 
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


def convert_edge_set_to_nxGraph(set_T: set) -> nx.Graph:
    """
    Converts the algorithms T set of edges (allocations) to 
    a Graph object.

    Note: Because of NetworkX edge settings inconsistency for edge 
          insertion we need to check each edge and it's opposite direction equally

    >>> agent1 = AdditiveAgent({'x':1, 'y':2, 'z':3}, name="agent1")
    >>> agent2 = AdditiveAgent({'x':3, 'y':2, 'z':1}, name="agent2")
    >>> test_set = set()
    >>> test_set.add((agent1, 'x'))
    >>> test_set.add((agent1, 'y'))
    >>> test_set.add((agent2, 'y'))
    >>> test_set.add((agent2, 'z'))
    >>> test_set.add((agent2, 'x'))
    >>> gr = convert_edge_set_to_nxGraph(test_set)
    >>> if (agent1, 'x') in gr.edges() or ('x', agent1) in gr.edges():
    ...     print("True")
    True
    >>> if (agent2, 'x') in gr.edges() or ('x', agent2) in gr.edges():
    ...     print("True")
    True
    >>> if (agent2, 'z') in gr.edges() or ('z', agent2) in gr.edges():
    ...     print("True")
    True
    """
    result_g = nx.Graph()
    for edge in set_T:
        result_g.add_node(edge[0])
        result_g.add_node(edge[1])
    for edge in set_T:
        result_g.add_edge(edge[0], edge[1])
    return result_g


def get_agent_fraction(allocation: FractionalAllocation, agent: AdditiveAgent, item) -> float:
    """
    Return an agents item fraction allocation part 
    in a FRactionalAllocation object


    >>> agent3 = AdditiveAgent({'x':1, 'y':-2, 'z':3}, name="agent3")
    >>> agent4 = AdditiveAgent({'x':3, 'y':2, 'z':-1}, name="agent4")
    >>> alloc = FractionalAllocation([agent3, agent4], [{'x':0.4, 'y':0, 'z':0.5},{'x':0.6, 'y':1, 'z':0.5}])
    >>> get_agent_fraction(alloc, agent4, 'x')
    0.6
    >>> get_agent_fraction(alloc, agent3, 'y')
    0
    """
    for i_agent, alloc_agent in enumerate(allocation.agents):
        if alloc_agent == agent:
            return allocation.map_item_to_fraction[i_agent][item]


def convert_FractionalAllocation_to_ValuationMatrix(allocation: FractionalAllocation) -> ValuationMatrix:
    """
    A helper function that converts a FractionalAllocation object to ValuationMatrix
    This function is used to prepare the matrix parameters for the linear solving process


    >>> agent1 = AdditiveAgent({"x": 1, "y": 2, "z": 4}, name="agent1")
    >>> agents = [agent1]
    >>> items_for_func ={'x','y','z'}
    >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':1.0, 'z':1.0}])
    >>> mat = convert_FractionalAllocation_to_ValuationMatrix(allocations)
    >>> mat[0,0]
    1
    >>> mat[0,1]
    2
    >>> mat[0][1]
    2
    >>> mat[0,2]
    4
    >>> mat
    [[1 2 4]]
    >>> mat.agent_value_for_bundle(0, [0, 1, 2])
    7
    >>> mat.agent_value_for_bundle(0, [0, 2])
    5
    >>> agent1 = AdditiveAgent({"x": 1, "y": 2, "z": 4}, name="agent1")
    >>> agent2 = AdditiveAgent({"x": 5, "y": -2, "z": 7}, name="agent2")
    >>> agents = [agent1, agent2]
    >>> items_for_func ={'x','y','z'}
    >>> allocations = FractionalAllocation(agents, [{'x':1.0,'y':0, 'z':1.0}, {'x':0,'y':1.0, 'z':0}])
    >>> mat = convert_FractionalAllocation_to_ValuationMatrix(allocations)
    >>> mat
    [[ 1  2  4]
     [ 5 -2  7]]
    >>> mat[1,0]
    5
    >>> mat[1][1]
    -2
    >>> mat[1,2]
    7
    >>> mat.agent_value_for_bundle(1, [0, 1, 2])
    10
    """
    matrix_valuation_list = []
    agent_index = 0
    for agent_valuations in allocation.map_item_to_fraction:
        agent_alloc_list = []
        for item, _ in agent_valuations.items():
            agent_alloc_list.append(allocation.agents[agent_index].value({item}))
        agent_index += 1
        matrix_valuation_list.append(agent_alloc_list)
    v_mat = ValuationMatrix(matrix_valuation_list)
    return v_mat


if __name__ == "__main__":
    # import doctest
    # (failures, tests) = doctest.testmod(report=True)
    # print("{} failures, {} tests".format(failures, tests))

    agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
    agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
    all_items  = {'a', 'b', 'c', 'd'}
    all_agents = [agent1, agent2]
    initial_allocation = FractionalAllocation(all_agents, [{'a':1.0,'b': 0.3,'c':0.5,'d':0.0},{'a':0.0,'b':0.7,'c':0.5,'d':1.0}])
    print(initial_allocation)
    pi = ParetoImprovement(initial_allocation, all_items)
    # pi.find_pareto_improvement().is_complete_allocation()
    print(pi.find_pareto_improvement())
