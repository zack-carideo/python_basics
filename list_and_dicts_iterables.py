###
## COMMON DICTIONARY AND LIST OPERATIONS
###
import numpy as np
import math
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




"""OVERVIEW OF MAP(https://realpython.com/python-map-function)

map(function, iterable[, iterable1, iterable2,..., iterableN])

    - function: the function that transforms each original item into a new (transformed) item. 
                Even though the Python documentation calls this argument function, it can be any Python callable. 
                This includes built-in functions, classes, methods, lambda functions, and user-defined functions.
   
Overview of map() 
    - map:  built-in function that allows you to process and transform all the items in an iterable without using an explicit for loop
            a technique commonly known as mapping. map() is useful when you need to apply a transformation function to each item in an iterable
            and transform them into a new iterable. map() takes a function object and an iterable (or multiple iterables) as arguments and returns 
            an iterator that yields transformed items on demand. Items in the new iterable are produced by calling the transformation function on 
            each item in the original iterable.

    - when to use map()
        - if you need to perform the same operation on all the items of an input iterable to build a new iterable (replaces for loop)

Advantages of map vs list 
    -  map() is written in C and is highly optimized, its internal implied loop can be more efficient than a regular Python for loop
    -  map() consumes minimal memory relative to a loop.  With a for loop, you need to store the whole list in your system’s memory. With map(), you get items on demand, and only one item is in your system’s memory at a given time.

Disadvantages of map 
    - map() operations can be easily replaced with list comprehension and geneartor  comprehension operations , and are more pythonic then map()

WARNINGS
    - The first argument to map() is a FUNCTION object, which means that you need to pass a function without calling it. That is, WITHOUT using a pair of parentheses.
    - After a iterable is mapped, if the output is not assigned back to a vairable, the mapped values are not retained in the original variable containing the map() call (i.e you cannot call list() on the mapped outtput more than 1 time without erasign the mapped valuse)
"""
# example function to map
def square(number):
    return number**2

# BASIC EXAMPLES OF MAP()
# map each value in iterable to a int and then map the function squared() to each value in transformed list 
# note: Since map() returns an iterator (a map object), you’ll need call list() so that you can exhaust the iterator and turn it into a list object. (i.e list(inter_of_ints))
iterable_1 = ["1","2","3","4","5","-56","9"]
iter_of_ints = list(map(int,iterable_1))
iter_of_abs_ints = list(map(abs,iter_of_ints))
iter_of_squared_ints = map(square , iter_of_abs_ints) #note: since you dont force the map generator to execute the map, you cannot use iter_of_quared_ints more than 1 time
iter_of_ints_lengths = map(len , iterable_1)


# USING LAMBDA WITH MAP 
# A common pattern that you’ll see when it comes to using map() is to use a lambda function as the first argument. 
# lambda functions are handy when you need to pass an expression-based function to map().
squared = list(map(lambda num: num**2, iter_of_ints))

# PROCSSING MULTIPLE INPUT ITERABLES WITH MAP()
# If you supply multiple iterables to map(), then the transformation function must take as many arguments as iterables you pass in. 
# Each iteration of map() will pass one value from each iterable as an argument to function. The iteration stops at the end of the shortest iterable.
# NOTE: This technique allows you to merge two or more iterables of numeric values using different kinds of math operations. 
iterable_2 = [1,2,3,4,99]
list(map(lambda x,y: x-y, iter_of_ints, iterable_2))
list(map(lambda x,y,z: ( x - y ) / z, iter_of_ints, iterable_2,iter_of_abs_ints))

#MAPPING A SINGLE ITERABLE TO A FUNCTION RETURNING MULTIPLE ITEMS 
def multi_return(integer_): 
    return integer_, integer_ + 10

list(map(lambda x: multi_return(x),iterable_2)) #lambda implementation 
list(map(multi_return, iterable_2)) #traditional implementation 


"""OVERVIEW OF FILTER with MAP
map().filter() : filter() is used with map() when you need to process an input iterable and return another iterable that results from filtering out unwanted values in the input iterable.

filter(function,iterable):
    - function: a BOOLEAN function returning True/False according to input data
    - iterable: any python iterable 
"""
#sample function to illistrate filter()
#we want to take square root of iterable values, but you must filter  neg numbers first becuase they do not have a sqrt
iterable_ = [1,-1,3,4,-33]

#boolean function used to tell map what to filter before applying sqrt
def is_positive(num_):
    return num_>=0

clean_sqrt = list(map(
                    math.sqrt
                    , filter(is_positive, iterable_)
                    ))



""" 
MAP + REDUCE()
- reduce() is used with map() when you need to apply a function to an iterable and reduce the output to a single value 
- reduce() will apply function to all the items in iterable and cumulatively compute a final value.

REDUCE is not used very often anymore , but very useful for understanding map/reduce frameworks 
"""

#ex calculating total size of all files in a directory 
import functools , operator, os, os.path 
files = os.listdir(os.path.expanduser("C:/Users/zjc10/Desktop/Projects/data/cfpb/"))
functools.reduce(operator.add, map(os.path.getsize, files))

"""STARMAP WITH TUPLE BASED ITERABLES 
Python’s itertools.starmap() makes an iterator that applies a function to the arguments obtained from an iterable of tuples 
and yields the results. It’s useful when you’re processing iterables that are already grouped in tuples.

The main difference between map() and starmap() is that the latter calls its transformation function using the unpacking operator (*)
to unpack each tuple of arguments into several positional arguments. So, the transformation function is called as function(*args) instead of function(arg1, arg2,... argN).

#visual of what starmap does, loops over elements in iterable, and yields a function with the iterable values unpacked and loaded ready to return result upon request
def starmap(function, iterable):
    for args in iterable:
        yield function(*args)
"""
from itertools import starmap
def raise2power(base,power):
    return base**power

tup_iter = [(1,1),(1,2),(2,1),(2,2)]
list(starmap(raise2power, tup_iter))

"""

END MAP OVERVIEW 

"""

