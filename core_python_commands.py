#find where python is installed 
where python

#upgrade package 
pip install numpy -U

#upgrade pip
python -m pip install --upgrade pip

#temporarly add python to system path (great if u dont have admin rights) 
set Path = C:\Program Files\Python37\python.exe;%Path%


#identify dependencies of any module (and the dependency versions) 
from pip._vendor import pkg_resources
_package_name = 'keras_bert'
_package = pkg_resources.working_set.by_key[_package_name]
print([str(r) for r in _package.requires()])

#list files in directory 
def list_files(dir_path):
    files = []
    for name in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, name)):
        files.append(name)
    return files

"""
sys.getsizeof(): inspects the size of an object
notice: both return the same information, one is a generator(yields one obs at a time when called), one is a list(saves all obs to memory)
"""
import sys
import random
_test_list = [random.randint(0,i) for i in range(1,100)]
_test_gen = (random.randint(0,i) for i in range(1,100))
print(sys.getsizeof(_test_list))
print(sys.getsizeof(_test_gen))

"""
cProfile.run() : generates a readout of an operations execution time and total calls 
"""
import cProfile
cProfile.run('sum(_test_list)') #list comprehension performance
cProfile.run('sum(_test_gen)') #generator comprehension performance



""" eval() , exec(),  compile() 

- eval(expression[, globals[, locals]])
    - function inputs 
        - expression(required): a string or compiled code input to be evaluated 
        - globals(optional): holds a dictionary that provides a global namespace to eval(),
                             it allows you to control which global names to use while evaluating the expression. 
                             you can also supply names(and values) that don't exist in the current global scope to eval()
        - locals(optional) : The main practical difference between globals and locals is that Python will automatically 
                             insert a "__builtins__" key into globals if that key doesn’t already exist. 
        
    - eval CANNOT evaluate python statements such as 
        - keyword-based compound statements 
            ex. eval("if x: print(x)") -> SyntaxError 
        - assignment statements 
            ex. eval("x = 30") -> Syntax Error 
        
    - general information 
        - allows you to evaluate arbitrary Python expressions from a string-based or compiled-code-based input.
        - has access to global and local object names, but does NOT have access to the nested scoped(non-locals) in the enclosing environment 
        - comprehensions (i.e. list, dict, etc...) ARE CONSIDERED EXPRESSIONS AND CAN BE EVALUATED USING EVAL() 
        - can NOT be used to evaluate compound statements(i.e. eval("if x: print(x)")  or assignment operations (i.e  eval("pi = 3.5"))
        - it is NOT possible to use keyword arguments with eval(), aka, if you want to pass a custom dictionary of locals() to eval() 
          but no globals you must provide an empty dic for globals() positional parameter
         
        

- eval() vs exec() : eval() can only execute or evaluate expressions, whereas exec() can execute any piece of python code 
    
- compilation steps when using eval() to evaluate a STRING based expression 
    1. Parse Expression 
    2. Compile it to bytecode
    3. Evaluate it as a python expression 
    4. return the result of evaluation 
    
- compilation steps when using eval() to evaluate a compiled code object created using compile() 
    1. Evalute the compiled code 
    2. Return the result of the evaluation 

- compile(source, filename, mode, flags=0, dont_inherit=False, optimize=- 1)
    - function inputs 
        - source:  holds the source code that you want to compile. This argument accepts normal strings, byte strings, and AST objects.
        - filename: gives the file from which the code was read. If you’re going to use a string-based input, then the value for this argument should be "<string>".
        - mode:  specifies which kind of compiled code you want to get. If you want to process the compiled code with eval(), then this argument should be set to "eval".
            
    - general information
        - compile is a built in function that can compile an input string into a code object or AST object that can be evaluated with eval() 
        - you can use compile() to supply code objects to eval() instead of normal strings
        - handy when you need to evaluate the same expression multiple times. precompile the expression and reuse the resulting bytecode on subsequent calls to eval().
    

- exec(object[, globals[, locals]])
    - 
    
- resources 
    - https://realpython.com/python-eval-function/
"""
#sample data for use in illustrating eval() and compile()
import pandas as pd
a = [1,2,3,4,5]
x = 3
zjc = 'zack'
df = pd.DataFrame([{'name':'zack','state':'NC'},{'name':'jack','state':'NY'}])

#1. basic use of eval() with STRING evaluation
#1.1: raw math, evaluate variable, using eval() in list comprehension
eval("2**7")
eval("sum(a)")
eval("[val_ + x for val_ in a]")

#1.2: using eval() with local and global parameters
eval("2+x+yyy",{},{"x":x, "yyy":43}) #using empty globals() dict and populated locals() dict() to limit scope of variables eval() can use during evaluation , note you can define new variables directly in locals() dict and use them in eval()


#2. Evaluating expressions with eval()
eval("x!=y",{},{"x":x,"y":3})
eval("x in a")

#3: using eval() with compile() to pre-compile code and evaluate with eval() (good when you have to evaluate a statement many times)
code = compile("5+3","<string>","eval")
eval(code)

#3.1: compile with error :
code = compile("d = 5+3","<string>","eval")  #ERROR, even when using compile you cannot pass assignment operations to eval
eval(code)

#4: using eval with general purpose expressions (ex showing how to launch firefox)
import subprocess
eval('subprocess.getoutput("firefox")')
eval("subprocess.getoutput('echo Hello, World')")

#5: using eval within a function to implement conditional statements that need to change dynamically
def func(a,b, condition):
    if eval(condition):
        return a+b
    return a-b

func(2,4, "a>b")


