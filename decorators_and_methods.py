import typing
import logging
import os
import sys
import time
import functools
import pandas as pd
from contextlib import contextmanager

#boolean flag to indicate if you want to run examples (only use if you are not importing things from this file)
run_examples = False

#logger for print to screen logging
logging.basicConfig(stream=sys.stdout,level=logging.INFO)
logger = logging.getLogger(__name__)

# code illustrating and implementing how to implement various python decorators and how to create a custom built decorators
'''
- table of contents
    -@property decorator
    -@contextmanager decorator
    -custom decorator to time a function @timeit


- definitions
    - decorator: decorator is a function that modify another functions behavior without affecting the
                              decorated functions core functionality. it is a function that takes a funciton
                              as input, and returns a function as output. a decorator adds functionality before 
                              and after the decorated function is executed, in essence expanding the decorated 
                              functions scope without actually altering the decorated function.



'''

###########################
# @property & _repr_ dunder#
###########################
'''
property method:
    - property(fget,fset,fdel,doc)
        - fget() : function for getting the attribtue value
        - fset() : function for setting the attribute value
        - fdel() : function for deleting the attribute value
        - doc()  : string containing documentation for the property method

    - provides a an interface to instance attributes and encapsulates instance attributes with properties
    - takes the get, set, and delete methods as arguments and returns an object of the property class
    - use the property decorator , not the method (more pythonic)

@property decorator:
    built in function that creates and returns a property object.


_repr_() dunder: python dunder method used to generate a string representation of a class object to enable replication(i.e. using eval())


'''


# example class with property object used to define what functions to call
# when someone access's, sets, or deletes  the 'name' attribute
# NOTE: the name attribute does not exist in the class , only __name exists , name
# is the property used to control how __name is set and retrieved.
class person:
    def __init__(self, name):
        self.__name = name

    # return a string representation of the class object when called (aka an uninitalized instance of person or an existing instance of person())
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items()])})"

    def setname(self, name):
        print('setname() called')
        self.__name = name

    def getname(self):
        print('getname() called')
        return self.__name

    def delname(self):
        print('delname() called')
        del self.__name

    # Set property to use get_name, set_name
    # and del_name methods
    name = property(getname, setname, delname)


# usage
# p1 = person('steve') # initalize
# p1.name = 'zack'     # set name attribute , which automatically calls setname (setter method)
# p1.name              # get name attribute, which automatically calls getname(getter method)
# del p1.name          # delete the internal __name


# EXAMPLE USING DECORATOR
class person:
    def __init__(self, name):
        self._name = name

    @property
    def getname(self):
        print('getname() called')
        return self._name

    @getname.setter
    def setname(self, name):
        print('setname() called')
        self._name = name

    @getname.deleter
    def delname(self):
        print('delname() called')
        del self._name


# usage
# p1 = person('steve') # initalize
# p1.name = 'zack'     # set name attribute , which automatically calls setname (setter method)
# p1.name              # get name attribute, which automatically calls getname(getter method)
# del p1.name          # delete the internal __name


##################
# @context_manager#
##################
"""
context manager: a protocol your object needs to follow so it can be used with the 'with' statment. 
key components(dunders) required to use 'with' statement on a class: __enter__, __exit__ 
Python will call these two methods at the appropriate times in the resource management cycle
Python calls __enter__ when execution enters the context of the with statement and it’s time to acquire the resource. 
When execution leaves the context again, Python calls __exit__ to free up the resource.

The with statement simplifies exception handling by encapsulating standard uses of try/finally statements in so-called Context Managers.
"""


# example class compatable with with statmenet
class ManagedFile:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.file = open(self.name, 'w')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


# example execution (very similar to the open() function)
# with ManagedFile('hello.txt') as f:
#    f.write('hello, world!')
#    f.write('bye now')

# same as above , but in function form leveraging @context_manager decorator
'''
In this case, managed_file() is a generator that first acquires the resource. 
Then it temporarily suspends its own executing and yields the resource so it can be used by the caller.
When the caller leaves the with context, the generator continues to execute so that any remaining clean 
up steps can happen and the resource gets released back to the system
'''


@contextmanager
def managed_file(name):
    try:
        f = open(name, 'w')
        yield f
    finally:
        f.close()


if run_examples:
    with managed_file('hello.txt') as f:
         f.write('hello, world!')
         f.write('bye now')
##############################
# CREATING YOUR OWN DECORATOR##
##############################

'''
create a decorated function to time how long it takes any function to execute and print to log 
1. start timer
2. execute the decorated function 
3. stop timer 
'''


def Timer(func: typing.Callable):
    '''
    Timer decorator
    a function to be used to time the execution of other functions
    '''

    @functools.wraps(func)

    # define the 'inner' function to be executed when the decorator(@Timer) is called(note it takes in 1 input, a function)
    def time_me(*args, **kwargs ) -> typing.Any:

        # create meta data about the function the decorator is applied to
        func_name = func.__name__

        # start the timer
        start = time.time()

        # execute the function being decorated (we use args , kwargs so the decorator can be used to evaluate any function and generalizes)
        value = func(*args, **kwargs)

        # note end time
        end = time.time()

        # write result to log
        logger.info(f'Operation {func_name} took {(end - start):.5f}s.')
        return value

    return time_me

# EXAMPLE
@Timer
def open_file(file_path):

    logger.info(os.path.splitext(file_path)[-1])
    if os.path.splitext(file_path)[-1] == '.parquet':
        df = pd.read_parquet(file_path)
    elif os.path.splitext(file_path)[-1] == '.csv':
        logger.info('read csv')
        df = pd.read_csv(file_path)
    elif os.path.splitext(file_path)[-1] == '.xlsx':
        logger.info('read xlsx')
        df = pd.read_xlsx(file_path)
    elif os.path.splitext(file_path)[-1] in ['.pkl','.pickle']:
        logger.info('read pickle')
        df = pd.read_pickle(file_path)
    else:
        raise Exception(f'i dont know how to read this file extension: {os.path.splitext(file_path)[-1]}')

    return df

if run_examples:
    #every time you run the funciton, it will be decorated with the @timer
    dd = open_file("C://Users//zjc10//Desktop//Projects//data//econ//Historic_Domestic.csv")