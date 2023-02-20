# utilities
# author: Charles Lee
# date: 2023.01.09

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy

def record_time(func):
    def inner_func(*args,**kwargs):
        time1 = time.time()
        result = func(*args,**kwargs)
        time2 = time.time()
        print("record_time: {:.4f}s".format(time2 - time1))
        return result 
    return inner_func




