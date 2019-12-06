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

## How to run
 

## Collected data
  - We conducted an experimental campaign recruiting participants through mailing lists and social media, and collected 742 samples from users in regions around the world. 
  - Adding to the platform code, we also publish the dataset with the results of our experiments. 
  - Logfiles are compiled in a JSON format, as shown below:
<p align="center">
  <img src="./Images/_logfile_format_.png">
</p>
