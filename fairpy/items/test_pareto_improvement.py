# #!python3

# """
# Main testing module for pareto_improvement.py
# Programmer: Shmuel Lavian
# Since:  2022-04
# """

# from fairpy.items.pareto_improvement import ParetoImprovement
# from fairpy.agents import AdditiveAgent
# from fairpy.items.allocations_fractional import FractionalAllocation
# import networkx as nx

# def test_case_1():
#     agent1= AdditiveAgent({"a": -100, "b": 10, "c": 50, "d": -100 ,"e": 70,"f": 100, "g": -300, "h": -40, "i": 30}, name="agent1")
#     agent2= AdditiveAgent({"a": 20, "b": 20, "c": -40, "d": 90 ,"e": -90,"f": -100, "g": 30, "h": 80, "i": 90}, name="agent2")
#     agent3= AdditiveAgent({"a": 10, "b": -30, "c": 30, "d": 40 ,"e": 180,"f": 100, "g": 300, "h": 20, "i": -90}, name="agent3")
#     agent4= AdditiveAgent({"a": -200, "b": 40, "c": -20, "d": 80 ,"e": -300,"f": 100, "g": 30, "h": 60, "i": -180}, name="agent4")
#     agent5= AdditiveAgent({"a": 50, "b": 50, "c": 10, "d": 60 ,"e": 90,"f": -100, "g": 300, "h": -120, "i": 180}, name="agent5")
#     list_of_agents_for_func = [agent1, agent2, agent3, agent4, agent5]
#     items_for_func = {'a','b','c','d','e','f','g','h','i'}
#     alloc_y_for_func = FractionalAllocation(list_of_agents_for_func, [
#         {'a':0.0,'b':1.0,'c':0.0,'d':0.0,'e':1.0,'f':1.0,'g':0.0,'h':0.0,'i':0.4},
#         {'a':0.0,'b':0.0,'c':0.0,'d':1.0,'e':0.0,'f':0.0,'g':0.0,'h':1.0,'i':0.0},
#         {'a':0.0,'b':0.0,'c':1.0,'d':0.0,'e':0.0,'f':0,'g':1.0,'h':0.0,'i':0.0},
#         {'a':0.0,'b':0.0,'c':0.0,'d':0.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0,'i':0.0},
#         {'a':1.0,'b':0.0,'c':0.0,'d':0.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0,'i':0.6}])
#     pi = ParetoImprovement(alloc_y_for_func, items_for_func)
#     assert pi.find_pareto_improvement().is_complete_allocation() == True
#     print(pi.find_pareto_improvement())


# if __name__ == "__main__":
#     test_case_1()