# Steam Game Analysis  
  
<img src="pictures/pipeline.png" width="350" align="right">
This is a team project to analyze steam games using Databricks. We created MongoDB Atlas and shard the data into three clusters which help improve our reading and writing data. Then we built a data pipeline to access 20GB+ raw data from Amazon S3 and transferred cleaned data into MongoDB Atlas. After we preprocess the data, we can load data from MongoDB Atlas and use Spark ML to applied Machine Learning Algorithm in the distributed system.  

-----
We have two different data:

* Steam Review data: Features about each review.  
<img src="pictures/review_info.png" width="350" align="right" >  

* Steam Games data: contains features about the game.
<img src="pictures/game_info.png" width="350" align="right" >
  

# User Segmentation  

# Predictive Models
Predict Voted up.....
## Logistic Regeression  

## Decision Tree  

## Random Forest  
We also built Random Forest to predict whether user will voted up for a game. Since we have 100k+ rows, it's hard to implement Random Forest on all features. 
## Comparing

# Game Recommendation
