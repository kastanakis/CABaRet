#!/usr/bin/env python
from abc import ABC, abstractmethod
from contentCatalogue import contentCatalogue
from cache import LRUCache

class RS(ABC):
    def __init__(self, sizeOfRecommendationList=20):
        '''
        Assigns the size of the recommendation list

        Input: sizeOfRecommendationList: integer
        '''
        self.sizeOfRecommendationList = sizeOfRecommendationList

    def characteristics(self):
        '''
        Output: returns specs of RS
        '''
        return 'Size of recommendation list: {}'.format(self.sizeOfRecommendationList)

    @abstractmethod
    def recommendationAlgorithm(self):
        ''' 
        This abstract method forces subclasses to implement this function
        '''
        pass


# TODO: rename this class as "mostRelatedRS", and returned the top N most related (where N=sizeOfRecommendationList); if there are more than M (>N) with the top relation (e.g. =1), then randomly sample N out of these M (again sample randomly, do not take the first N; see comments in other files about random choices)
class mostRelatedRS(RS):
    def __init__(self, sizeOfRecommendationList=20):
        RS.__init__(self, sizeOfRecommendationList)

    def recommendationAlgorithm(self, id, contentCatalogue):
        ''' 
        Returns a recommendation list based on the provided ID
        The RL occurs from the content catalogue and the content matrix
        (see contentCatalogue class to understand extraction of related) .
        This Rec. Alg. knows nothing about network and caches -> cache
        agnostic.

        Input: ID
        Output: Recommendation List
        '''
        return [x[0] for x in contentCatalogue.relatedContents(
            id)[0:self.sizeOfRecommendationList]]


class cacheAwareRS(RS):
    def __init__(self, sizeOfRecommendationList=20):
        RS.__init__(self, sizeOfRecommendationList)

    def removeDuplicates(self, seq):
        '''
        Removes duplicates from a list
        '''
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def CABaRet(self, D, W, id, contentCatalogue, cachedContents):
        # TODO: this implementation of CABaRet is simplistic and incorrect: (a) what id D=3? (b) what if less than N contents are found in the cache? how many recommendations will be returned? (c) if more than N contents are found in the cache, do your code guarantees that the most related cached are returned?
        ''' 
        Returns a recommendation list based on the provided ID
        The RL occurs from the content catalogue and the content matrix
        (see contentCatalogue class to understand extraction of related) .
        This Rec. Alg. is bfs-related and cache aware, thus knows the cached contents, 
        which are a small number of contents extracted from the content catalogue
        to simulate a cache, using bfs traversing.


        Input: Seed ID, Width, Depth, Content Catalogue, Cached Contents 
        Output: Recommendation List
        '''
        # Collect 1st depth related contents
        L = list()
        relatedContents = [x[0]
                           for x in contentCatalogue.relatedContents(id)[0:W]]
        for item in relatedContents:
            if cachedContents.isInCache(item):
                L.append(item)
        if D == 2:
            # Collect 2nd depth related contents
            for item in relatedContents:
                _relatedContents = [x[0] for x in contentCatalogue.relatedContents(item)[
                    0:W]]
                for _item in _relatedContents:
                    if cachedContents.isInCache(_item):
                        L.append(_item)
        for item in relatedContents:
            # Fill the rest of the RL with the remaining related contents on the seed id
            # (apart from the ones that are already included in RL from the previous step)
            L.append(item)

        # Remove duplicates
        L = self.removeDuplicates(L)
        return L[0:self.sizeOfRecommendationList]

    def recommendationAlgorithm(self, D=2, W=50, id='', contentCatalogue=list(), cachedContents=list()):
        ''' 
        Returns a recommendation list based on the provided ID
        The RL occurs from the content catalogue and the content matrix
        (see contentCatalogue class to understand extraction of related) .
        This Rec. Alg. knows cached contents -> cache aware.

        Input: ID
        Output: Recommendation List
        '''

        return self.CABaRet(D, W, id=id, contentCatalogue=contentCatalogue,
                            cachedContents=cachedContents)


def main():
    _contentCatalogue = contentCatalogue(size=5)
    _contentCatalogue.initializeContentCatalogue(
        ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff"])
    # _contentCatalogue.createRandomContentMatrix()
    _contentCatalogue.loadContentMatrix_JSON(
        r'JointCachingRecommendations\dynamic_caching\Simulator\contentMatrix.json')
    print(_contentCatalogue.characteristics())
    
    _cache = LRUCache(size=1)
    _cache.initializeCache("eeee")

    rs1 = mostRelatedRS()  # by default the size of RL is 20
    print(rs1.recommendationAlgorithm("aaaa", _contentCatalogue))

    rs2 = cacheAwareRS()
    print(rs2.recommendationAlgorithm(2, 50, "aaaa", _contentCatalogue, _cache))


if __name__ == "__main__":
    main()
