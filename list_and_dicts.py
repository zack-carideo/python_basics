###
## COMMON DICTIONARY AND LIST OPERATIONS
###
import numpy as np
from typing import List


#example lists for use
l1 =[1,2,np.nan, 43, 44]
l2 = ['a','b','c','d','nan',np.nan]


#count total nans in list
def count_nans(in_lst: List ) -> int:
    """
    inputs:
        in_lst : single dimension list to count nan occurrences from
    ex:
        count_nans(l1)
    """
    return np.count_nonzero(np.isnan(in_lst))


#generate random combinations of values in a list
#a value cannot be used twice or be paired with itself
import random
def rand_list_map(_list):
    out_map = {}
    used = []

    while len(used) != len(_list) :
        giver = random.choice([val for val in _list if val not in used])
        reciever = random.choice(_list)

        if giver!=reciever and giver not in used:
            out_map[giver] = reciever
            used.append(giver)

    return out_map




#sort a list of tuples
def sort_tup_list(loftups: List, sort_on: int = 0):
    """
    loftups: list of tuples to be sorted
    sort_on: index of the element in each tuple to sort the list on
    """
    return sorted(loftups, key=lambda x: x[sort_on])

