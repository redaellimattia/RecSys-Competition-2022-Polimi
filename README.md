# RecSys Competition 2022 - Polimi

This repository contains the code used during the [Recommender Systems 2022 Competition](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi) hosted on Kaggle, which is part of the course of Recommender Systems at the Polytechnic University of Milan.

The application domain is TV shows recommendation. The datasets provided contain both interactions of users with TV shows, as well as features related to the TV shows. The main goal of the competition is to discover which items (TV shows) a user will interact with. More info can be found on [Kaggle](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi).

- Final Leaderboard position: 25/99

## 🗒 Introduction

Most of the models come from the [src](/src) folder which is a clone of the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), please refer to the repository for the installation guide.  
Our aim was to find the best scoring model by tuning and combining them with the techniques seen during the course.

The [data_util.py](/utils/data_util.py) has been used to ease the handling of the dataset.

## 🎯 Final Model

After testing out a lot of approaches, **RP3Beta** seemed the best recommender for this competition.  
Also **SLIM ElasticNet** (after a lot of tuning 🥵) produced good recommendations.

In the end an hybrid of the two has been used.
Here's the list of links to the notebooks used for the final model:

- [RP3Beta](/notebooks/RP3_Beta_HT.ipynb): This notebook implements RP3Beta and performs hyper tuning of its parameters.

- [SLIM ElasticNet](/notebooks/SLIM_ElasticNet_HT.ipynb): This notebook implements RP3Beta and performs hyper tuning of its parameters.

- [Hybrid](/notebooks/Hybrid_RP3Beta_SLIMElasticNet_HT.ipynb): This notebook performs a linear combination of the two recommenders, using their best parameters.
