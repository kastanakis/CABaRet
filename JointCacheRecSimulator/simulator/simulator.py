#!/usr/bin/env python
from contentCatalogue import contentCatalogue
from cache import LRUCache
from recommendationSystem import mostRelatedRS, cacheAwareRS
from user import user
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

CONTENT_CATALOGUE_SIZE = 10000
RECOMMENDATION_LIST_SIZE = 10

def zipf_pmf(lib_size, expn):
    '''
    Applies the probability mass function for zipf

    Input: size of content catalogue, exponent
    Output: list of probabilities that sum up to 1
    '''
    K = lib_size
    p0 = np.zeros([K])
    for i in range(K):
        p0[i] = ((i+1)**(-expn))/np.sum(np.power(range(1, K+1), -expn))
    return p0


def main():
    # Load a content catalogue
    _contentCatalogue = contentCatalogue(size=CONTENT_CATALOGUE_SIZE)
    _contentCatalogue.loadContentMatrix_CSV(
        r'C:\Users\kastanakis\Documents\GitHub\JointCachingRecommendations\Results\contentCatalogue.csv')
    # print(_contentCatalogue.characteristics())

    # Load optimal cached contents generated from optimization algorithm
    _cache = LRUCache()
    out = None
    with open(r'C:\Users\kastanakis\Documents\GitHub\JointCachingRecommendations\Results\optimizationAlgoResults\N10_C10_Pzipf-1_greedy_cache_opt.json', 'r') as f:
        out = json.load(f)
    _cache.initializeCache(out['cache_vector'])
    print(_cache.characteristics())

    # Ensure that all cached contents exist in content catalogue
    assert(len(list(set(_contentCatalogue.contents).intersection(
        set(_cache.contents)))) == len(_cache.contents))

    # Generate a recommender
    rs1 = mostRelatedRS(sizeOfRecommendationList=RECOMMENDATION_LIST_SIZE)
    rs2 = cacheAwareRS(sizeOfRecommendationList=RECOMMENDATION_LIST_SIZE)

    popularities = zipf_pmf(RECOMMENDATION_LIST_SIZE, 1.0)
    print(popularities)
    # Run CABaRet for each content in the content catalogue.
    # Measure average CHR, sum of popularities VS cache size
    
    allCHRsums = 0
    allPOPsums = 0
    counter = 0
    for item in _contentCatalogue.contents:
        print(counter)
        counter += 1
        relList2 = rs2.recommendationAlgorithm(
            2, 5, item, _contentCatalogue, _cache)
        CHRsum = 0
        POPsum = 0
        indexInPopularitiesList_recList = 0
        indexInPopularitiesList_contentCatalogue = 0
        for rel in relList2:
            if _cache.isInCache(rel):
                CHRsum += popularities[indexInPopularitiesList_recList]
                POPsum += _contentCatalogue.popularity[indexInPopularitiesList_contentCatalogue]
            indexInPopularitiesList_recList += 1
            indexInPopularitiesList_contentCatalogue += 1

        allCHRsums += CHRsum
        allPOPsums += POPsum
    print(allCHRsums / CONTENT_CATALOGUE_SIZE)
    print(allPOPsums / CONTENT_CATALOGUE_SIZE)
    
    

if __name__ == "__main__":
    main()
