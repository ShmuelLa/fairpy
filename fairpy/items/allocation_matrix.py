import numpy as np

from fairpy.items.allocations_fractional import FractionalAllocation
from fairpy.agents import AdditiveAgent


class AllocationMatrix:
    """
    Allocation matrix class will be used to convert FractionalAllocation objects to 
    a matrix type object to be used in cvxpy iterations

    Programmer: Shmuel Lavian
    Since:  2022-04

    >>> agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
    >>> agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
    >>> all_items  = {'a', 'b', 'c', 'd'}
    >>> all_agents = [agent1, agent2]
    >>> initial_allocation = FractionalAllocation(all_agents, [{'a':0.0,'b': 0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
    >>> a_mat = AllocationMatrix(initial_allocation)
    >>> a_mat[0, 0]
    0.0
    >>> a_mat[1][0]
    1.0
    >>> a_mat[1][1]
    0.7
    """

    def __init__(self, allocation: FractionalAllocation):
        """
        initialize an allocation matrix from a fractional allocation object
        this class is used for the pareto improvement linear algorithm phase

        Args:
            allocation (FractionalAllocation): a fractional allocation to convert
        """
        self.allocation = allocation
        self.num_of_agents = len(allocation.map_item_to_fraction)
        self.num_of_objects = len(allocation.map_item_to_fraction[0].keys())
        matrix_allocation_list = []
        for agent_allocations in allocation.map_item_to_fraction:
            matrix_allocation_list.append(list(agent_allocations.values()))
        self._a_mat = np.array(matrix_allocation_list)


    def __getitem__(self, key) -> float:
        """
        Returns the specific item allocation valuation in [x, y] type

        Args:
            key (_type_): the specific key to return

        Returns:
            _type_: the specific value of the allocation
        """
        if isinstance(key, tuple):
            return self._a_mat[key[0], key[1]]  # agent's allocation for a single object
        else:
            return self._a_mat[key]             # agent's allocation for all objects

    
    def agents(self):
        return range(self.num_of_agents)

    def objects(self):
        return range(self.num_of_objects)

if __name__ == "__main__":
    import doctest
    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))

    # agent1 = AdditiveAgent({"a": 10, "b": 100, "c": 80, "d": -100}, name="agent1")
    # agent2 = AdditiveAgent({"a": 20, "b": 100, "c": -40, "d": 10}, name="agent2")
    # all_items  = {'a', 'b', 'c', 'd'}
    # all_agents = [agent1, agent2]
    # initial_allocation = FractionalAllocation(all_agents, [{'a':0.0,'b': 0.3,'c':1.0,'d':0.0},{'a':1.0,'b':0.7,'c':0.0,'d':1.0}])
    # a_mat = AllocationMatrix(initial_allocation)
    # print(a_mat[0, 0])

