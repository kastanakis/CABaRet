# CABaRet Experimental Testbed

## Motivation
We implemented an experimental platform to collect data from real users. 
Our  goals are:
- monitor real users' behavior in VoD ecosystems
- evaluate the Cache Hit Ratio (CHR) that can be achieved in practice by a cache-aware recommender system
- produce insights regarding Quality of Recommendations (QoR)

## Overview
The platform is built on top of the YouTube video service: it streams videos through the YouTube service, 
and uses the YouTube API to retrieve recommendations and related contents.

## Experiment Session
The UI is designed to accommodate our experiments and a screenshot is shown below.
For the back-end, we assume that a list of cached video IDs is available at the time of the experiment, and we use the YouTube API to embed a YouTube video player in our platform and serve video contents to the participants of the experiment.
Finally, the recommendation module is implemented, using CABaRet as the recommendation algorithm.

<p align="center">
  <img src="./Images/_experiment_screenshot_.png">
</p>

## Experiment Setup
- Region
  - We offer as options a subset of the regions provided by the YouTube API. 
  - We selected 7 representative regions (different continents, diverse demographics, available video data).

- Initial list of videos
  - For every region, we retrieve from the YouTube API the list of 50 top trending videos.
  - We randomly select 20 of them (for the selected region) to present to the user.

- Caching
  - We compiled a list of 500 videos IDs that are assumed to be cached
  - We consider a different list per region:
    - In each list, we select to first include the top 50 trending videos in this region. 
    - Then, for each of these 50 videos, we request its 50 related videos provided by YouTube API. 
    - From these 2500 (50 × 50) total videos, we add in the list the 450 videos with the higher number of views (“most popular”).
    
- List of recommendations
The list of the 5 recommendations given to the user when watching a video are generated by CABaRet.
CABaRet algorithm is shown in the below scheme:
<p align="center">
  <img src="./Images/_algorithm_module_.png">
</p>

- Collected data


