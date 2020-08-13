#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
import random
import string
from scipy.stats import zipf
from itertools import chain
import json


class contentCatalogue():
    def __init__(self, size=1000):
        '''
        Assigns the size and constructs an empty list of contents. Constructs
        an empty list for popularities of contents (probabilities). Constructs an
        empty content matrix as a list.

        Input: contents: list, popularity: list, contentMatrix: list, size: int
        '''
        self.size = size
        self.contents = list()
        self.popularity = list()
        self.contentMatrix = list()

    def characteristics(self):
        '''
        Output: returns specs of content catalogue
        '''
        return 'Content Catalogue Size: {}\nContent Catalogue Popularity:\n{}\nContent Catalogue:\n{}\nContent Catalogue Relations:\n{}'.format(self.size, self.popularity, self.contents, self.contentMatrix)

    def randomSingleContentGenerator(self, stringLength=8):
        """Generate a random string of letters and digits """
        lettersAndDigits = string.ascii_letters + string.digits
        return ''.join(random.choice(lettersAndDigits) for i in range(stringLength))

    def randomMultipleContentsGenerator(self):
        """Generate a list of random strings of letters and digits """
        contents = list()
        for i in range(0, self.size):
            contents.append(self.randomSingleContentGenerator())

        assert(len(contents) == self.size)
        return contents

    def getNrandomElements(self, list, N):
        '''
        Returns random elements from a list

        Input: list and num of items to be returned
        Output: list of N random items
        '''
        return random.sample(list, N)

    def getContents(self):
        '''
        Output: returns contents as a list
        '''
        return self.contents

    def getContentsLength(self):
        '''
        Output: returns contents size
        '''
        return len(self.contents)

    def zipf_pmf(self, lib_size, expn):
        '''
        Returns a probability mass function (list of probabilities summing to 1.0) for the Zipf distribution with the given size "lib_size" and exponent "expn"

        Input: size of content catalogue, exponent
        Output: list of probabilities that sum up to 1
        '''
        K = lib_size
        p0 = np.zeros([K])
        for i in range(K):
            p0[i] = ((i+1)**(-expn))/np.sum(np.power(range(1, K+1), -expn))
        return p0

    def setContentsPopularity(self, distribution='zipf', a=0.78):
        '''
        Sets the popularity of contents given a distribution (zipf by default) .

        Input: distribution, exponent
        Output: vector of probabilities that correspond to the content catalogue
        '''
        if distribution == 'zipf':
            prob = self.zipf_pmf(self.getContentsLength(), a)
            return prob
        else:
            raise Exception('Distribution \'' + distribution +
                            '\' not implemented yet')

    def initializeContentCatalogue(self, contents):
        '''
        Set the content catalogue for the first time in a list format

        Input: strings in a list
        '''
        self.contents = contents
        self.popularity = self.setContentsPopularity()

    def symmetrize(self, a):
        '''
        Forces symmetricity in a content matrix

        Input: a matrix
        Output: a symmetric matrix provided from the original
        '''
        return np.tril(a) + np.tril(a, -1).T

    def createRandomContentMatrixBinary(self, symmetric=True, numOfRelations=10, outputForm='dataframe'):
        '''
        TODO: Fix commentary in this piece of code
        '''
        numOfContents = self.getContentsLength()
        contentMatrix = np.zeros((numOfContents, numOfContents))
        idx = np.random.rand(numOfContents, numOfContents).argsort(1)[
            :, :numOfRelations]
        contentMatrix[np.arange(numOfContents)[:, None], idx] = 1
        print(contentMatrix)
        # if symmetric:
        #     contentMatrix = self.symmetrize(contentMatrix)
        # print(contentMatrix)

        # for row in contentMatrix:
        #     if(len(row) != numOfRelations):
        #         print(row)
        # # print(contentMatrix)
        # for i in range(numOfContents):
        #     for j in range(numOfContents):
        #         if i == j and contentMatrix[i][j] == 1:
        #             indexesOfZeros = np.argwhere(
        #                 contentMatrix[i] == 0).tolist()
        #             contentMatrix[i][j] = 0
        # for i in range(numOfContents):
        #     for j in range(numOfContents):
        #         # print(i, j)
        #         indexesOfOnesCurrentNodeRow = np.argwhere(
        #             contentMatrix[i] == 1).tolist()
        #         # print(len(indexesOfOnesCurrentNodeRow))
        #         while len(indexesOfOnesCurrentNodeRow) < numOfRelations:
        #             randomChoiceOfIndex = random.choice(
        #                 indexesOfOnesCurrentNodeRow)[0]
        #             indexesOfOnesRelatedNodesRow = np.argwhere(
        #                 contentMatrix[randomChoiceOfIndex] == 1).tolist()
        #             if len(indexesOfOnesRelatedNodesRow) < numOfRelations:
        #                 contentMatrix[i][randomChoiceOfIndex] = 1
        #                 contentMatrix[randomChoiceOfIndex][i] = 1

        # assert symmetricity
        # assert(np.allclose(contentMatrix, contentMatrix.T))
        self.contentMatrix = contentMatrix

        # Return in a specific format (list or df)
        if outputForm == 'dataframe':
            names = [_ for _ in self.getContents()]
            df = pd.DataFrame(contentMatrix, index=names, columns=names)
            return df
        return contentMatrix

    def loadContentMatrix_JSON(self, url):
        '''
        Loads an item based NxN content matrix (IDs in rows/columns).

        Input: url of content matrix
        '''
        out = None
        with open(url, 'r') as f:
            out = json.load(f)
        self.contentMatrix = np.array(out)

    def loadContentMatrix_CSV(self, url):
        '''
        Loads an item based NxN content matrix (IDs in rows/columns).
        Also initializes the content catalogue with the given column names

        Input: url of content matrix as a CSV file
        '''
        data = pd.read_csv(url, delimiter='\t')
        self.initializeContentCatalogue(list(data.columns)[1:])
        data = [list(x[1:]) for x in data.to_numpy()]
        self.contentMatrix = np.array(data)

    def relatedContents(self, id):
        '''
        Returns all non zero relations to a given ID
        Relations are extracted from a content matrix (cm)


        Input: id
        Output: related contents list
        '''
        # extract all relations from content matrix
        candidateRelated = self.contentMatrix[self.contents.index(id)]
        # print(len(candidateRelated))
        # extract all non zero relations from the above list - just indexes
        indexesOfPositiveRelations = np.argwhere(candidateRelated == 1)
        # print(len(indexesOfPositiveRelations))

        # make the above indexes a single list, for easier reference
        indexesOfPositiveRelations = list(
            chain.from_iterable(indexesOfPositiveRelations))
        # dereference the indexes => acquire a list of related contents
        related = [self.contents[i] for i in indexesOfPositiveRelations]
        toReturn = []
        # Return also the relation weight for each related content
        for rel in related:
            toReturn.append(
                (rel, candidateRelated[self.contents.index(rel)]))

        # Return items sorted in descending relevance (most relevant item in first position)
        return sorted(toReturn, key=lambda x: x[1], reverse=True)


def main():
    N = 10  # number of relations
    W = 5
    MP = 10000
    r = contentCatalogue(size=10000)
    r.initializeContentCatalogue(r.randomMultipleContentsGenerator())
    # Set numOfRelations equal to the number of relations you want each content to have with all others
    r.createRandomContentMatrixBinary(symmetric=False, numOfRelations=N)

    # Get content catalogue
    names = [_ for _ in r.getContents()]
    df = pd.DataFrame(r.contentMatrix, index=names, columns=names)
    df.to_csv(r'JointCachingRecommendations\Results\contentCatalogue.csv', sep='\t')

    # Get top most popular contents
    mostPopularContents = r.getContents()[0:MP]
    # print(r.relatedContents(mostPopularContents[0]))
    with open(r'JointCachingRecommendations\Results\mostPopular.json', 'w') as f:
        json.dump(mostPopularContents, f, indent=4)

    # Get depth 1 related
    depth1 = {}
    for popular in mostPopularContents:
        depth1[popular] = [x[0] for x in r.relatedContents(popular)[0:W]]
        # print(len(depth1[popular]))
    with open(r'JointCachingRecommendations\Results\dataSet_depth1_width50.json', 'w') as f:
        json.dump(depth1, f, indent=4)

    # Get depth 2 related
    depth1values = list(
        set([item for sublist in depth1.values() for item in sublist]))
    depth2 = {}
    for item in depth1values:
        depth2[item] = [x[0] for x in r.relatedContents(item)[0:W]]
    with open(r'JointCachingRecommendations\Results\dataSet_depth2_width50.json', 'w') as f:
        json.dump(depth2, f, indent=4)


if __name__ == "__main__":
    main()
