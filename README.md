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
![Image]()
(more details in~\secref{sec:experiment-session}). For the back-end, we assume that a list of cached video IDs is available at the time of the experiment (see~\secref{sec:experiments-setup}), and we use the YouTube API to embed a YouTube video player in our platform and serve video contents to the participants of the experiment. Finally, the recommendation module is implemented as described in~\secref{sec:overview}, using \ourAlgo as the recommendation algorithm.

## Experiment Setup
- Region
- Initial list of videos
- Caching
- List of recommendations
- Collected data


