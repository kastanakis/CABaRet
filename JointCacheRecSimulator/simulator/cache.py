#!/usr/bin/env python
from abc import ABC, abstractmethod

class Cache(ABC):
    def __init__(self):
        '''
        Constructs an empty list and initializes size of cache
        '''
        self.contents = list()
        self.size = 0

    def characteristics(self):
        ''' 
        Output: returns specs of cache
        '''
        return 'Cache Size: {}\nCached Contents:\n{}'.format(self.size, self.contents)

    def getContents(self):
        ''' 
        Output: returns cached contents as a list
        '''
        return self.contents

    def isInCache(self, key):
        ''' 
        Input: requested ID 
        Output: returns 1/0 for hit/miss respectively
        '''
        return key in self.contents

    @abstractmethod
    def initializeCache(self, contents):
        ''' 
        This abstract method forces subclasses to implement this function
        '''
        pass

    @abstractmethod
    def getItem(self, key):
        ''' 
        This abstract method forces subclasses to implement this function
        '''
        pass


class LRUCache(Cache):
    def __init__(self):
        Cache.__init__(self)
        self.contents = list()

    def initializeCache(self, contents):
        ''' 
        Set the cached contents for the first time in a list format

        Input: strings in a list
        '''
        self.contents = contents
        self.size = len(contents)

    def getItem(self, key):
        ''' 
        Update policy of an LRU cache.

        Input: requested ID
        Output: ID, cache hit/miss
        '''
        hit = 0
        if self.isInCache(key):
            self.contents.remove(key)
            hit = 1
        else:
            self.contents.pop(0)
        self.contents.append(key)
        return key, hit


def main():
    # c = Cache() - throw exception because you cant instantiate an abstract class
    c = LRUCache()
    print(c.characteristics())
    print(c.getContents())

    c.initializeCache(["a", "b", "c", "d", "e", "f"])
    print(c.getContents())
    print(c.isInCache("f"))
    # print(c.getItem("f"))
    # print(c.getContents())
    # print(c.getItem("f"))
    # print(c.getContents())
    # print(c.getItem("q"))
    # print(c.getContents())
    # print(c.getItem("rr"))
    # print(c.getContents())
    # print(c.getItem("rr"))
    # print(c.getContents())
    # print(c.getItem("q"))
    # print(c.getContents())


if __name__ == "__main__":
    main()
