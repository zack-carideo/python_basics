#
# LIST OF EXAMPLE USES FOR MANIPULATING NUMPY ARRAYS
#

import numpy as np

# example arrays for demonstration
d1_array = np.array([1, 2, 3])
d2_array = np.array([[1, 2, 3], [4, 5, 6]])
d3_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


"""
np.prod(): return the product of array elements over a given axis (also excepts tuple of elements)
"""

# by default np.prod() will calculate the product of all elements in the input object (regardless of dimension)
print(f'accepts tuple {np.prod((1, 2, 3))} or array {np.prod(d1_array)}')
print(f'if no axis is specified {np.prod(d2_array)} everything is multipled -> 1*2*3*4*5*6')
print(f'if no axis is specified {np.prod(d3_array)} everything is multipled -> 1*2*3*4*5*6*7*8*9')

# axis=1 , multiples the elements of each dimension of the array and returns an array of same len but 1 value per array
# axis=0 , multiples across dimensions (first element from each dim is multipled, then 2nd, then 3rd)
print(f'specifying axis=1 : {np.prod(d3_array, axis=1)}')
print(f'specifying axis=0 : {np.prod(d3_array, axis=0)}')

"""
np.cumprod(): returns the cumulative product of elements along a given axis 
"""



"""
np.unique(): identify all unique elements in a numpy array and return a freq count
"""
unique, counts = np.unique(d3_array, return_counts=True)
