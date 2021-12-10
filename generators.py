"""
GENERATORS
    - Generators simplify creation of iterables. Anything that can be done with a generator can be done by implementing iterator protocol. 
    - Generators require lesser lines of code than solving the problem with iterators and iterables.
    - Generators are functions having an yield keyword. Any function which has “yield” in it is a generator.
    - Calling a generator function creates an iterable. Since it is an iterable so it can be used with iter() and with a for loop.
    - Generator takes care of creating the iterable. It also takes care of creating the underlying iterator. And next() of this iterator() is such that 
    it returns each ‘yield’ value of generator one after the other. When there is no more ‘yield’ in the generator function then this iterator raises StopIteration.

"""


"""
Implementing a for loop using a while statement and a generator 
Note: we leverage StopIteration exception to break out of the while loop once the generated has exhausted all inputs
"""
letters = ['a','b','c','d','e']
letter_gen = (letter for letter in ['a','b','c','d','e'])
it = iter(letters)
it = iter(letter_gen)

while True:
    try:
        letter = next(it)
    except StopIteration:
        break
    print(letter)