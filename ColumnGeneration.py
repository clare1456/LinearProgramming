"""
Column Generation
author: Charles Lee
date: 2023.01.08
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

class ColumnGeneration():
    def __init__(self, graph, RLMP=None, SP=None, column_pool=None):
        self.graph = graph
        # link models given or build them
        if RLMP is None:
            model_handler = ModelHandler(graph)
            self.RLMP = model_handler.build_RLMP_model(graph)
        else:
            self.RLMP = RLMP 
        if SP is None:
            model_handler = ModelHandler(graph)
            self.SP = model_handler.build_SP_model(graph)
        else:
            self.SP = SP
        # link column_pool or build it
        if column_pool is None:
            self.column_pool = {}
            # add initial routes into pool
            init_routes = self.RLMP._init_routes
            for i, route in enumerate(init_routes):
                self.column_pool[f"y[{i}]"] = route
        else:
            self.column_pool = column_pool

        # algorithm part
        self.duals_of_RLMP = {}
        self.RLMP_obj = np.inf # record result
        self.SP_obj = np.inf # use to check termination 
        self.EPS = 1e-6 # use to check termination
        # display part
        self.cg_iter_cnt = 0
        self.OutputFlag = 1
    
    def solve_RLMP_and_get_duals(self):
        """
        solve RLMP and get dual values 
        """
        self.RLMP.optimize()
        if self.RLMP.Status != 2:
            return 0 # infeasible
        for cons in self.RLMP.getConstrs():
            cons_name = cons.ConstrName 
            self.duals_of_RLMP[cons_name] = cons.Pi
        self.RLMP_obj = self.RLMP.ObjVal
        return 1 # feasible

    def solve_SP(self):
        # update objective with duals
        obj = 0.0
        for i in range(self.graph.nodeNum):
            for j in self.graph.feasibleNodeSet[i]:
                var_name = f"x[{i},{j}]"
                cons_name = f"R{j}"
                coef = self.graph.disMatrix[i, j]
                if j > 0:
                    coef -=  self.duals_of_RLMP[cons_name]
                obj += coef * self.SP.getVarByName(var_name)
        self.SP.setObjective(obj, GRB.MINIMIZE) 
        # optimize SP
        self.SP.optimize()
        self.SP_obj = self.SP.ObjVal

    def get_columns_and_add_into_RLMP(self):
        """ get column from SP model and add into RLMP """
        new_route, route_length, new_column = self.get_column_from_SP()
        self.add_column_into_RLMP(new_route, route_length, new_column)
        
    def get_column_from_SP(self):
        """
        belonging to get_columns_and_add_into_RLMP()  
        get column from solved SP model
        """
        new_route = [0]
        route_length = 0
        new_column = np.zeros(self.graph.nodeNum)
        current_i = 0
        new_column[0] = 1 # coefficient of vehicleNum constraint
        while True:
            for j in self.graph.feasibleNodeSet[current_i]:
                var_name = f"x[{current_i},{j}]"
                var_val = round(self.SP.getVarByName(var_name).X)
                if var_val == 1:
                    route_length += self.graph.disMatrix[current_i, j]
                    new_route.append(j)
                    new_column[j] = 1 # coefficient of customer pass constraint
                    current_i = j
                    break
            if current_i == 0:
                break
        return new_route, route_length, new_column

    def add_column_into_RLMP(self, new_route, route_length, new_column):
        """
        belonging to get_columns_and_add_into_RLMP()  
        add the column generated by SP model to RLMP
        """
        # update column pool
        new_column_name = "y[{}]".format(len(self.column_pool))
        self.column_pool[new_column_name] = new_route
        # update RLMP
        new_RLMP_column = gp.Column(new_column, self.RLMP.getConstrs())
        self.RLMP.addVar(obj=route_length, vtype=GRB.CONTINUOUS, column=new_RLMP_column, name=new_column_name)
        self.RLMP.update()

    def solve_final_RMP(self):
        # convert RLMP into RMP
        self.RMP = self.RLMP.copy()
        for var in self.RMP.getVars():
            var.vtype = 'B'
        # optimize RMP
        self.RMP.optimize()
        return self.RMP

    # output methods
    def output_info(self):
        if self.OutputFlag:
            print("CG iter {}: RLMP_obj = {}, SP_obj = {}, Columns_num = {}".format(self.cg_iter_cnt, self.RLMP_obj, self.SP_obj, len(self.column_pool)))

    def get_routes(self):
        """
        get routes according to solved RMP
        """
        # get each routes
        routes = []
        for var in self.RMP.getVars():
            var_name = var.VarName
            if abs(var.X - 1) < self.EPS:
                route = self.column_pool[var_name]
                routes.append(route)
        return routes

    # main funtions
    def column_generation(self):
        """
        pure column generation   
        """
        while True:
            # solve RLMP and get duals
            is_feasible = self.solve_RLMP_and_get_duals()
            if is_feasible != 1:
                return 0
            # solve SP
            self.solve_SP()
            # record information
            self.cg_iter_cnt += 1
            self.output_info()
            # break if can't improve anymore
            if self.SP_obj >= -self.EPS:
                return 1
            # get columns and add into RLMP
            self.get_columns_and_add_into_RLMP()

    def run(self):
        """
        column generation and get integer result  
        """
        is_feasible = self.column_generation()
        if is_feasible:
            self.solve_final_RMP()
            return self.get_routes()
        else:
            print("Model Infeasible")
            return None
    
class ColumnGenerationWithLabeling(ColumnGeneration):
    def __init__(self, graph, RLMP=None, column_pool=None):
        super().__init__(graph, RLMP=RLMP, SP=[], column_pool=column_pool) # no need to build SP
        self.labeling = Labeling(graph, outputFlag=False)
        # self.labeling = Labeling(graph, select_num=100, early_stop=1) # fast version

    def solve_SP(self):
        """
        solve subproblem with labeling algorithm 
        """
        # set dual values
        duals = [self.duals_of_RLMP[f"R{i}"] for i in range(self.graph.nodeNum)]
        self.labeling.set_dual(duals)
        routes, objs = self.labeling.run()
        self.labeling_routes = routes
        self.labeling_objs = objs
        self.SP_obj = min(objs) if len(objs) > 0 else np.inf

    def get_columns_and_add_into_RLMP(self):
        """
        get columns from labeling and add into RLMP 
        """
        # get routes
        routes = self.labeling_routes
        # add routes into RLMP
        for route in routes: 
            # calculate route_length
            route_length = 0
            new_column = np.zeros(self.graph.nodeNum)
            new_column[0] = 1
            for i in range(1, len(route)):
                route_length += self.graph.disMatrix[route[i-1], route[i]]
                new_column[route[i]] = 1
            self.add_column_into_RLMP(route, route_length, new_column)

if __name__ == "__main__":
    file_name = "solomon_100\R101.txt"
    graph = GraphTool.Graph(file_name)
    # alg = ColumnGeneration(graph) # result: optimal 828.936, time 359.5s
    alg = ColumnGenerationWithLabeling(graph) # result: optimal 828.936, time 68.0s
    routes = alg.run()
    obj = graph.evaluate(routes)
    print("obj = {}".format(obj))
    graph.render(routes)