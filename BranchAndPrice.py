"""
Branch and Price
author: Charles Lee
date: 2022.11.07
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB

import GraphTool
from ModelHandler import ModelHandler
from Labeling import Labeling
from ColumnGeneration import *

class BPNode():
    def __init__(self, graph, RLMP, SP, global_column_pool):
        self.graph = graph
        self.global_column_pool = global_column_pool
        self.RLMP = RLMP.copy()
        self.SP = SP
        self.CG_alg = ColumnGeneration(self.graph, self.RLMP, self.SP, self.global_column_pool)
        self.initialize()
    
    def initialize(self):
        """
        initialize coefficients when created fresh new
        """
        # model part
        self.SP_must_include = []
        self.SP_cant_include = []
        # algorithm part
        self.local_LB = -np.inf
        self.IP_obj = np.inf
        self.LP_obj = -np.inf
        self.EPS = 1e-6 
        self.is_feasible = False
        self.is_integer = False
        self.inf_var_list = []
        self.x_sol = {}
        self.x_int_sol = {}
        # display part
        self.depth = 0
        self.has_showed_way_of_opt = False
        self.way_of_opt = "---"
        self.prune_info = "---"
    
    def generate(self):
        subNode = BPNode(self.graph, self.RLMP, self.SP, self.global_column_pool)
        subNode.local_LB = self.LP_obj
        subNode.depth = self.depth + 1
        subNode.SP_must_include = self.SP_must_include.copy()
        subNode.SP_cant_include = self.SP_cant_include.copy()
        return subNode
    
    def solve_and_update(self):
        """ solve and check feasibility """
        self.is_feasible = self.column_generation()
        if self.is_feasible == 0:
            return
        """ update x_sol and round to x_int_sol """
        vars = self.RLMP.getVars()
        self.is_integer = True
        for var in vars:
            var_name = var.VarName
            var_val = var.X
            self.x_sol[var_name] = var_val
            self.x_int_sol[var_name] = round(var_val)
            # check integer
            if (abs(round(var_val) - var_val) > self.EPS):
                self.is_integer = False
                self.inf_var_list.append(var_name)
        """ update LP / IP """
        self.local_LB = self.RLMP.ObjVal
        self.LP_obj = self.RLMP.ObjVal
        if self.is_integer:
            self.way_of_opt = "By Simplex"
            obj = 0
            for var in vars:
                var_name = var.VarName
                obj += self.x_int_sol[var_name] * var.obj
            self.IP_obj = obj
        else:
            # solve RMP to get a integer solution as UB
            self.solve_final_RMP_and_update_IPobj()
    
    def column_generation(self):
        tmp_disMatrix = self.graph.disMatrix.copy() # test program
        self.set_SP()
        is_feasible = self.CG_alg.column_generation() 
        self.reset_SP()
        assert np.sum(self.graph.disMatrix - tmp_disMatrix) == 0
        return is_feasible
              
    def set_SP(self):
        """ update must_include / cant_include constraints """
        for pair in self.SP_must_include:
            pi, pj = pair
            for j in range(self.graph.nodeNum):
                if j == pj:
                    continue
                var_name = f"x[{i},{j}]"
                self.SP.getVarByName(var_name).setParam("UB", 0) #! check
        for pair in self.SP_cant_include:
            pi, pj = pair
            var_name = f"x[{i},{j}]" 
            self.SP.getVarByName(var_name).setParam("UB", 0) #! check
        self.SP.update()

    def reset_SP(self):
        """ reset must_include / cant_include constraints """
        for pair in self.SP_must_include:
            pi, pj = pair
            for j in range(self.graph.nodeNum):
                if j == pj:
                    continue
                var_name = f"x[{i},{j}]"
                self.SP.getVarByName(var_name).setParam("UB", 1) 
        for pair in self.SP_cant_include:
            pi, pj = pair
            var_name = f"x[{i},{j}]" 
            self.SP.getVarByName(var_name).setParam("UB", 1) 
        self.SP.update()

    def solve_final_RMP_and_update_IPobj(self):
        # solve RMP
        self.RMP = self.CG_alg.solve_final_RMP()
        # update IP_obj if feasible
        if self.RMP.Status == 2:
            self.IP_obj = self.RMP.ObjVal
            self.way_of_opt = "By RMP"
    
    def get_columns_from_Labeling_and_add(self):
        # create tmp_graph, update must_include / cant_include constraints
        tmp_graph = deepcopy(self.graph)
        for pair in self.SP_must_include:
            pi, pj = pair
            for j in range(tmp_graph.nodeNum):
                if j==pj:
                    continue
                tmp_graph.disMatrix[pi][j] = np.inf
        for pair in self.SP_cant_include:
            pi, pj = pair
            tmp_graph.disMatrix[pi][pj] = np.inf
        # create labeling algorithm, set duals and solve
        alg = Labeling(tmp_graph)
        duals = np.zeros(tmp_graph.nodeNum)
        for di in range(tmp_graph.nodeNum):
            cons_name = f"R{di}"
            cons = self.RLMP.getConstrByName(cons_name)
            duals[di] = cons.Pi
        alg.set_dual(duals)
        alg.run()
        routes = alg.best_routes
        objs = alg.best_objs
        min_obj = min(objs)
        # add routes into RLMP
        col_num = self.graph.nodeNum
        for route in routes: 
            # calculate route_length
            route_length = 0
            new_column = np.zeros(self.graph.nodeNum)
            for i in range(1, len(route)):
                route_length += tmp_graph.disMatrix[route[i-1], route[i]]
                new_column[i] = 1
            self.add_column_into_RLMP(route, route_length, new_column)
        print(f"CG_iter {self.cg_iter_cnt}: RMPob{self.RLMP.ObjVal}, min_obj={self.SP.ObjVal}")
        return min_obj
        
class BPNodeWithLabeling(BPNode):
    def __init__(self, graph, RLMP, global_column_pool):
        self.graph = graph
        self.global_column_pool = global_column_pool
        self.RLMP = RLMP.copy()
        self.CG_alg = ColumnGenerationWithLabeling(self.graph, self.RLMP, self.global_column_pool)
        self.initialize()

    def generate(self):
        subNode = BPNodeWithLabeling(self.graph, self.RLMP, self.global_column_pool)
        subNode.local_LB = self.LP_obj
        subNode.depth = self.depth + 1
        subNode.SP_must_include = self.SP_must_include.copy()
        subNode.SP_cant_include = self.SP_cant_include.copy()
        return subNode

    def set_SP(self):
        """ update must_include / cant_include constraints """
        self.change_dict = {} # save changed data for reset_SP()
        for pair in self.SP_must_include:
            pi, pj = pair
            for i in self.graph.availableNodeSet[pj]:
                if i!=pi and pj!=0 and (i, pj) not in self.change_dict:
                    self.change_dict[i,pj] = self.graph.disMatrix[i][pj]
                    self.graph.disMatrix[i][pj] = np.inf
            for j in self.graph.feasibleNodeSet[pi]:
                if j!=pj and pi!=0 and (pi, j) not in self.change_dict:
                    self.change_dict[pi,j] = self.graph.disMatrix[pi][j]
                    self.graph.disMatrix[pi][j] = np.inf
        for pair in self.SP_cant_include:
            pi, pj = pair
            if pj in self.graph.feasibleNodeSet[pi] and (pi,pj) not in self.change_dict:
                self.change_dict[pi,pj] = self.graph.disMatrix[pi][pj]
                self.graph.disMatrix[pi][pj] = np.inf

    def reset_SP(self):
        """ reset must_include / cant_include constraints """
        for pair, dist in self.change_dict.items():
            self.graph.disMatrix[pair] = dist

class BranchAndPrice():
    def __init__(self, graph, is_labeling=1):
        # graph info
        self.graph = graph
        self.is_labeling = is_labeling
        self.model_handler = ModelHandler(graph)
        self.global_column_pool = {}
        # build and set RLMP, SP
        self.RLMP = self.set_RLMP_model(graph)
        if self.is_labeling == 0:
            self.SP = self.set_SP_model(graph)    
        # set strategies
        self.branch_strategy = "max_inf"
        self.search_strategy = "best_LB_first"
        # algorithm part
        self.node_list = []
        self.global_LB = -np.inf
        self.global_UB = np.inf
        self.EPS = 1e-6
        # display parament
        self.iter_cnt = 0
        self.Gap = np.inf
        self.fea_sol_cnt = 0
        self.BP_tree_size = 0
        self.branch_var_name = ""

    # basic functions
    def set_RLMP_model(self, graph):
        RLMP = self.model_handler.build_RLMP_model(graph)
        for route in RLMP._init_routes:
            var_name = "y[{}]".format(len(self.global_column_pool))
            self.global_column_pool[var_name] = route
        return RLMP
    
    def set_SP_model(self, graph):
        SP = self.model_handler.build_SP_model(graph) 
        return SP
    
    def root_init(self):
        if self.is_labeling:
            self.root_node = BPNodeWithLabeling(self.graph, self.RLMP, self.global_column_pool)
        else:
            self.root_node = BPNode(self.graph, self.RLMP, self.SP, self.global_column_pool)
        # self.node_list.append(self.root_node)
        self.root_node.solve_and_update()
        self.global_LB = self.root_node.LP_obj
        self.global_UB = self.root_node.IP_obj
        self.incumbent_node = self.root_node
        self.current_node = self.root_node
        if self.root_node.is_integer == False:
            self.branch(self.root_node)

    def search(self):
        """ best_LB_first: choose the node with best LB to search """
        best_node_i = 0
        if self.search_strategy == "best_LB_first":
            min_LB = np.inf
            for node_i in range(len(self.node_list)):
                LB = self.node_list[node_i].local_LB
                if LB < min_LB:
                    min_LB = LB
                    best_node_i = node_i
        best_node = self.node_list.pop(best_node_i)
        self.global_LB = min_LB # update global_LB
        return best_node
    
    def branch(self, node):
        # get flow of each arc
        flow_matrix = np.zeros((self.graph.nodeNum, self.graph.nodeNum))
        vars = node.RLMP.getVars()
        for var in vars:
            var_val = var.X
            var_name = var.VarName
            if var_val > 0:
                route = self.global_column_pool[var_name]
                for j in range(1, len(route)):
                    pi = route[j-1]
                    pj = route[j]
                    flow_matrix[pi][pj] += var_val
        # max_inf: choose the arc farthest to integer to branch 
        best_i = best_j = 0
        if self.branch_strategy == "max_inf":
            max_inf = -np.inf
            for pi in range(self.graph.nodeNum):
                for pj in self.graph.feasibleNodeSet[pi]:
                    if flow_matrix[pi, pj] > 1: #! skip flow more than 1
                        continue
                    if flow_matrix[pi, pj] < self.EPS:
                        continue
                    cur_inf = abs(flow_matrix[pi, pj] - round(flow_matrix[pi, pj]))
                    if cur_inf > max_inf:
                        max_inf = cur_inf
                        best_i = pi
                        best_j = pj
        # branch on the chosen variable
        self.branch_var_name = f"x[{best_i},{best_j}]"

        ## 1. branch left, must include xij
        leftNode = node.generate()
        ### delete routes constains i or j, but xij != 1
        for var in vars:
            var_name = var.VarName
            route = self.global_column_pool[var_name]
            if best_i not in route and best_j not in route:
                continue
            elif best_i in route and best_j in route:
                skip_flag = False
                for i in range(len(route)-1):
                    if route[i] == best_i and route[i+1] == best_j:
                        skip_flag = True
                        break
                if skip_flag:
                    continue
            elif (best_i == 0) and (best_i in route and best_j not in route):
                continue
            elif (best_j == 0) and (best_j in route and best_i not in route):
                continue
            var_name = var.VarName
            left_var = leftNode.RLMP.getVarByName(var_name)
            left_var.ub = 0 # make var_i invalid
        ### add into must include list
        leftNode.SP_must_include.append([best_i, best_j])
        self.node_list.append(leftNode)
        self.BP_tree_size+=1

        ## 2. branch right, cant include xij
        rightNode = node.generate()
        ### delete routes with xij == 1
        for var in vars:
            var_name = var.VarName
            route = self.global_column_pool[var_name]
            for i in range(len(route)-1):
                if route[i] == best_i and route[i+1] == best_j:
                    var_name = var.VarName
                    right_var = rightNode.RLMP.getVarByName(var_name)
                    right_var.ub = 0 # make var_i invalid
        ### add into cant include list
        rightNode.SP_cant_include.append([best_i, best_j])
        self.node_list.append(rightNode)
        self.BP_tree_size+=1

    # output functions
    def display_MIP_logging(self):
        """
        Show the MIP logging.

        :param iter_cnt:
        :return:
        """
        self.end_time = time.time()
        if (self.iter_cnt <= 0):
            print('|%6s  |' % 'Iter', end='')
            print(' \t\t %1s \t\t  |' % 'BB tree', end='')
            print('\t %10s \t |' % 'Current Node', end='')
            print('    %11s    |' % 'Best Bounds', end='')
            print(' %8s |' % 'incumbent', end='')
            print(' %5s  |' % 'Gap', end='')
            print(' %5s  |' % 'Time', end='')
            print(' %6s |' % 'Feasible', end='')
            print('     %10s      |' % 'Branch Var', end='')
            print()
            print('| %4s   |' % 'Cnt', end='')
            print(' %5s |' % 'Depth', end='')
            print(' %8s |' % 'ExplNode', end='')
            print(' %10s |' % 'UnexplNode', end='')
            print(' %4s |' % 'InfCnt', end='')
            print('    %3s   |' % 'Obj', end='')
            print('%10s |' % 'PruneInfo', end='')
            print(' %7s |' % 'Best UB', end='')
            print(' %7s |' % 'Best LB', end='')
            print(' %8s |' % 'Objective', end='')
            print(' %5s  |' % '(%)', end='')
            print(' %5s  |' % '(s)', end='')
            print(' %8s |' % ' Sol Cnt', end='')
            print(' %7s  |' % 'Max Inf', end='')
            print(' %7s  |' % 'Max Inf', end='')
            print()
        if(self.incumbent_node == None):
            print('%2s' % ' ', end='')
        elif(self.incumbent_node.way_of_opt == 'By Rouding' and self.incumbent_node.has_showed_way_of_opt == False):
            print('%2s' % 'R ', end='')
            self.incumbent_node.has_showed_way_of_opt = True
        elif (self.incumbent_node.way_of_opt == 'By Simplex' and self.incumbent_node.has_showed_way_of_opt == False):
            print('%2s' % '* ', end='')
            self.incumbent_node.has_showed_way_of_opt = True
        # elif (self.current_node.has_a_int_sol_by_heur == True and incumbent_node.has_showed_heu_int_fea == False):
        #     print('%2s' % 'H ', end='')
        #     self.incumbent_node.has_showed_heu_int_fea = True
        else:
            print('%2s' % ' ', end='')

        print('%3s' % self.iter_cnt, end='')
        print('%10s' % self.current_node.depth, end='')
        print('%9s' % self.iter_cnt, end='')
        print('%12s' % len(self.node_list), end='')
        if (len(self.current_node.inf_var_list) > 0):
            print('%11s' % len(self.current_node.inf_var_list), end='')
        else:
            if (self.current_node.RLMP.status == 2):
                print('%11s' % 'Fea Int',
                        end='')  # indicates that this is a integer feasible solution, no variable is infeasible
            elif (self.current_node.RLMP.status == 3):
                print('%11s' % 'Inf Model',
                        end='')  
            else:
                print('%11s' % '---',
                        end='')  
        if (self.current_node.RLMP.status == 2):
            print('%12s' % round(self.current_node.RLMP.ObjVal, 2), end='')
        else:
            print('%12s' % '---', end='')
        print('%10s' % self.current_node.prune_info, end='')
        print('%12s' % round(self.global_UB, 2), end='')
        print('%10s' % round(self.global_LB, 2), end='')
        if(self.incumbent_node == None):
            print('%11s' % '---', end='')
        else:
            print('%11s' % round(self.incumbent_node.IP_obj, 2), end='')
        if (self.Gap != '---'):
            print('%9s' % round(100 * self.Gap, 2), end='%')
        else:
            print('%8s' % 100 * self.Gap, end='')
        print('%8s' % round(self.end_time - self.start_time, 0), end='s')
        print('%9s' % self.fea_sol_cnt, end=' ')
        print('%14s' % self.branch_strategy, end='')
        print('%9s' % self.branch_var_name, end='')
        print()

    def display_result(self):
        self.CPU_time = time.time() - self.start_time
        print('\n')
        if len(self.node_list) == 0:
            print("Unexplored node list empty")
        else:
            print("Global LB and UB meet")
        print("Branch and bound terminates !!!")
        print("\n\n ------------ Summary ------------")
        print("Incumbent Obj: {}".format(self.incumbent_node.IP_obj))
        print("Gap: {}%".format(round(self.Gap * 100) if self.Gap < np.inf else 'inf')  )
        print("BB tree size: {}".format(self.BP_tree_size))
        print("CPU time: {}s".format(self.CPU_time))
        print(" --------- Solution  --------- ")
        for key in self.incumbent_node.x_int_sol.keys():
            if self.incumbent_node.x_int_sol[key] == 1:
                print('{} = {}: {}'.format(key, self.incumbent_node.x_int_sol[key], self.global_column_pool[key]))

    def get_routes(self):
        self.incumbent_node.CG_alg.solve_final_RMP()
        return self.incumbent_node.CG_alg.get_routes()

    # main functions
    def run(self):
        self.start_time = time.time()
        """ initalize the node """
        self.root_init() # solve root node and update global_LB/UB and branch
        """ branch and bound """
        while len(self.node_list) > 0 and self.global_LB < self.global_UB:
            """ search part """
            self.current_node = self.search() # get a node from node_list

            """ solve and prune """
            # prune1: By Bnd
            if self.current_node.local_LB >= self.global_UB: 
                self.current_node.prune_info = 'By Bnd'
            else:
                # prune2: By Inf
                self.current_node.solve_and_update()
                if not self.current_node.is_feasible:            
                    self.current_node.prune_info = 'By Inf'
                else:
                    # prune3: By Opt
                    incum_update = False # record whether incumbent updated
                    if self.current_node.is_integer:                 
                        self.fea_sol_cnt += 1
                        if self.current_node.IP_obj < self.global_UB: # update best solution
                            self.global_UB = self.current_node.IP_obj
                            self.incumbent_node = self.current_node
                            incum_update = True
                            if (self.global_UB < np.inf):
                                self.Gap = (self.global_UB - self.global_LB) / self.global_UB
                        self.current_node.prune_info = 'By Opt'

            """ branch part """
            if self.current_node.prune_info == '---': # if not been pruned
                self.branch(self.current_node)
        
            """ display logging """
            self.end_time = time.time()
            self.display_MIP_logging()
                
            self.iter_cnt += 1

        if len(self.node_list) == 0:
            self.global_LB = self.global_UB
        self.Gap = (self.global_UB - self.global_LB) / self.global_UB
        self.display_MIP_logging()
        """ display result """
        self.display_result()

        routes = self.get_routes()
        return routes

if __name__ == "__main__":
    file_name = "solomon_100\R101.txt"
    graph = GraphTool.Graph(file_name)
    alg = BranchAndPrice(graph) # result: R101 49s optimal 1642.87
    routes = alg.run()
    graph.evaluate(routes, show=True)
    graph.render(routes)






