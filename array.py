import numpy as np


class Array:

    def __init__(self, sizeOfVector):
        self.D = sizeOfVector
        self.array = np.zeros(self.D)
        self.memory = np.zeros(self.D)
        # Dictionary is used to hold the information of item before encoding.
        self.myDict = {}

    def __encode(self, item):
        """ Encode then return the encoded vector.
            Also add the encoded vector to the dictionary
        """
        self.myDict[item] = np.random.choice([-1, 1], self.D)
        return self.myDict.get(item)

    def __sim(self, item):
        if item in self.myDict:
            return np.dot(self.array, self.myDict.get(item)) / self.D
        return 0

    def add(self, item):
        self.array += self.__encode(item)

    def add_multiple(self, item, repeat):
        self.array += repeat * self.__encode(item)

    def remove(self, item):
        if item in self.myDict:
            self.array -= self.myDict.get(item)

    def contains(self, item):
        val = self.__sim(item)
        return 0.95 < val < 1.05

    def count(self, item):
        return self.__sim(item)

    def test(self):
        """ Function to show that a random vector that is not added to the
            array will return value close to zero.
        """
        return np.dot(self.array, np.random.choice([-1, 1], self.D)) / self.D

    # def __len__(self):

    # def traversal(self):


    # def setItem(self, index, itemToSet):


if __name__ == '__main__':
    # Array emulated with HD-Computing can ignore data types! Super cool

    arr = Array(10000)
    arr.add('dog')
    arr.add(1)
    arr.add(True)
    arr.add_multiple('cat', 3)  # Add cat 3 times?

    print(arr.contains('dog'))  # Should print True
    print(arr.contains(True))   # Should print True
    print(arr.contains('cat'))  # Should print True
    print(arr.count('cat'))     # Should print close to 3
    print(arr.test())           # Should print close to 0

    # print(len(arr))           # Should be six?





