# code illustrating and implementing how to implement various python decorators and custom built decorators


###########
#@property#
###########
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

'''

# example class with property object used to define what functions to call
# when someone access's, sets, or deletes  the 'name' attribute
#NOTE: the name attribute does not exist in the class , only __name exists , name
#is the property used to control how __name is set and retrieved.
class person:
    def __init__(self, name):
        self.__name=name
    def setname(self, name):
        print('setname() called')
        self.__name=name
    def getname(self):
        print('getname() called')
        return self.__name
    def delname(self):
        print('delname() called')
        del self.__name
    # Set property to use get_name, set_name
    # and del_name methods
    name=property(getname, setname, delname)

#usage
#p1 = person('steve') # initalize
#p1.name = 'zack'     # set name attribute , which automatically calls setname (setter method)
#p1.name              # get name attribute, which automatically calls getname(getter method)
#del p1.name          # delete the internal __name


#EXAMPLE USING DECORATOR
class person:
    def __init__(self, name):
        self.__name=name

    @property
    def getname(self):
        print('getname() called')
        return self._name

    @getname.setter
    def setname(self, name):
        print('setname() called')
        self._name=name

    @getname.deleter
    def delname(self):
        print('delname() called')
        del self._name

#usage
#p1 = person('steve') # initalize
#p1.name = 'zack'     # set name attribute , which automatically calls setname (setter method)
#p1.name              # get name attribute, which automatically calls getname(getter method)
#del p1.name          # delete the internal __name




