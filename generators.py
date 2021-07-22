"""
GENERATORS
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