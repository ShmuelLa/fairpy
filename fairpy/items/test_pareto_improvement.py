#!python3

"""
Main testing module for pareto_improvement.py

Programmer: Shmuel Lavian
Since:  2022-04
"""

from fairpy.items.pareto_improvement import ParetoImprovement
from fairpy.agents import AdditiveAgent
from fairpy.items.allocations_fractional import FractionalAllocation


def test_linear_programming_values():
    agent1 = AdditiveAgent({"a": 1, "b": 1, "c": 5}, name="agent1")
    agent2 = AdditiveAgent({"a": -1, "b": 6, "c": -3}, name="agent1")
    agent3 = AdditiveAgent({"a": -6, "b": -2, "c": -3}, name="agent1")

    list_of_agents_for_func = [agent1]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a': 1.0, 'b': 1.0, 'c': 1.0}])
    assert ParetoImprovement.linear_programming_solve_optimal_value(alloc_y_for_func)[0] == False
    list_of_agents_for_func = [agent2]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a': 1.0, 'b': 1.0, 'c': 1.0}])
    assert ParetoImprovement.linear_programming_solve_optimal_value(alloc_y_for_func)[0] == False
    list_of_agents_for_func = [agent3]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a': 1.0, 'b': 1.0, 'c': 1.0}])
    assert ParetoImprovement.linear_programming_solve_optimal_value(alloc_y_for_func)[0] == False


def test_linear_prog_dict():
    agent1 = AdditiveAgent({"a": 1, "b": 1, "c": 5}, name="agent1")
    agent2 = AdditiveAgent({"a": -1, "b": 6, "c": -3}, name="agent1")
    agent3 = AdditiveAgent({"a": -6, "b": -2, "c": -3}, name="agent1")

    list1 = [agent1]
    list2 = [agent2]
    list3 = [agent3]

    test_lists = [list1, list2, list3]

    for list in test_lists:
        alloc_y_for_func = FractionalAllocation(list, [{'a': 1.0, 'b': 1.0, 'c': 1.0}])
        fPO = ParetoImprovement.linear_programming_solve_optimal_value(alloc_y_for_func)
        # None value means the iteration finished correctly and left no other options
        for k, v in fPO.linear_agents_fragments.items():
            assert v == None
        for k, v in fPO.linear_agent_fragments.items():
            assert v == None


def test_single_agent_with_negatives_or_positives():
    agent1 = AdditiveAgent({"a": 1, "b": 2, "c": 3}, name="agent1")
    agent2 = AdditiveAgent({"a": -1, "b": -2, "c": -3}, name="agent1")
    list_of_agents_for_func = [agent1]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
    fpo_alloc = ParetoImprovement.find_pareto_improvement(alloc_y_for_func)
    assert fpo_alloc.is_complete_allocation() == True
    assert fpo_alloc.agents == list_of_agents_for_func
    assert fpo_alloc.agents[0] == list_of_agents_for_func[0]
    assert fpo_alloc.agents[0] == agent1
    assert fpo_alloc.agents[0].value() == 6

    list_of_agents_for_func = [agent2]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
    fpo_alloc = ParetoImprovement.find_pareto_improvement(alloc_y_for_func)
    assert fpo_alloc.is_complete_allocation() == True
    assert fpo_alloc.agents == list_of_agents_for_func
    assert fpo_alloc.agents[0] == list_of_agents_for_func[0]
    assert fpo_alloc.agents[0] == agent2
    assert fpo_alloc.agents[0].value() == -6


def test_single_agent_with_mixed_values():
    agent3 = AdditiveAgent({"a": -1, "b": 2, "c": -3}, name="agent1")
    list_of_agents_for_func = [agent3]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
    fpo_alloc = ParetoImprovement.find_pareto_improvement(alloc_y_for_func)
    assert fpo_alloc.is_complete_allocation() == True
    assert fpo_alloc.agents == list_of_agents_for_func
    assert fpo_alloc.agents[0] == list_of_agents_for_func[0]
    assert fpo_alloc.agents[0] == agent3
    assert fpo_alloc.agents[0].value() == -2


def test_single_agent_with_zero():
    agent4 = AdditiveAgent({"a": -1, "b": 0, "c": -3}, name="agent1")
    list_of_agents_for_func = [agent4]
    alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [{'a':1.0,'b':1.0, 'c':1.0}])
    fpo_alloc = ParetoImprovement.find_pareto_improvement(alloc_y_for_func)
    assert fpo_alloc.is_complete_allocation() == True
    assert fpo_alloc.agents == list_of_agents_for_func
    assert fpo_alloc.agents[0] == list_of_agents_for_func[0]
    assert fpo_alloc.agents[0] == agent4
    assert fpo_alloc.agents[0].value() == -4


def test_complex_agent_case():
    agent1 = AdditiveAgent({"a": 1, "b": 1, "c": 1}, name="agent1")
    agent2 = AdditiveAgent({"a": 1, "b": 1, "c": 1}, name="agent2")
    agent3 = AdditiveAgent({"a": 1, "b": 1, "c": 1}, name="agent3")
    all_agents = [agent1, agent2, agent3]
    f_alloc = FractionalAllocation(all_agents, [{"a": 0.2, "b": 0.3, "c": 0.5},{"a": 0.4, "b": 0.1, "c": 0.5},{"a": 0.4, "b": 0.1, "c": 0.0}])
    fpo_alloc = ParetoImprovement.find_pareto_improvement(f_alloc)
    assert fpo_alloc.agents == all_agents
    assert fpo_alloc.agents[0].value() == 0.5 or fpo_alloc.agents[0].value() == 0.2 or fpo_alloc.agents[0].value() == 0.3
    assert fpo_alloc.agents[1].value() == 0.4 or fpo_alloc.agents[1].value() == 0.5
    assert fpo_alloc.agents[2].value() == 0.4

