# Steam Game Analysis  
  
<img src="pictures/pipeline.png" width="350" align="right">This is a team project to analyze steam games using Databricks. We created MongoDB Atlas and shard the data into three clusters which help improve our reading and writing data. Then we built a data pipeline to access **20GB+** raw data from Amazon S3 and transferred cleaned data into MongoDB Atlas. After we preprocess the data, we can load data from MongoDB Atlas and use **Spark ML** to applied Machine Learning Algorithm in the distributed system.  

-----
We have two different data:

* Steam Review data: Features about each review.  

| Factor | DataType | Detail |
|--------|--------|--------|
| SteamID| Numeric | Steam User ID|
|.......

  
<img src="pictures/review_info.png" width="350" >  

* Steam Games data: contains features about the game.
<img src="pictures/game_info.png" width="350" >
  

# User Segmentation  

# Predictive Models
Predict Voted up.....
## Logistic Regeression  

## Decision Tree  

## Random Forest  
We also built Random Forest to predict whether a user will vote up for a game. Since we have 100k+ rows, it's hard to implement Random Forest on all features. So we need to select the feature and use feature engineering to get more predictive features. So we choose `"appid"`,`"platforms"`,`"num_games_owned"`,`"num_reviews"`,`"developer"`,`"price"`,`"publisher"`,`"playtime_at_review"` and `"playtime_forever"` to predict `"voted_up"`. Since the review also contains lots of information, we used **vaderSentiment** a package in Python that can convert text into the sentimental score. By defining a UDF function(user define function), we convert review context into the sentimental score in Spark Dataframe.   

We built Random Forest and used cross-validation to search for our best model.

```python
rf = RandomForestClassifier()
evaluator = BinaryClassificationEvaluator().setLabelCol("label").setMetricName("areaUnderPR")
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [10,15]).addGrid(rf.maxBins, [6000]).addGrid(rf.numTrees,[10,15]).build()
 
cv = CrossValidator(estimator=rf, 
                    evaluator=evaluator, 
                    numFolds=3, 
                    estimatorParamMaps=paramGrid)
...
```
After fitting the model, we can get feature importance from Random Forest. The result shows that sentimental score is the model important feature in the model. Playtime, Developer, and Price are also important to predict Voted up. Then we tested the best model on the validation set, the accuracy is about 0.97.

<img src="pictures/feature_importance_rf.png" width="500">



## Comparing
| Model | Time | Detail |
|--------|--------|--------|
| Logistic Regeression| ... | ...|
| Decision Tree| ... | ...|
| Random Forest| ... | ...|




# Game Recommendation
Commercial success of modern games hinges on player satisfaction and retention. So we did collaborative filtering recommendation via alternative least squares (ALS) algorithm. This Spark model only accepts user-item matrix for now (Year 2022), so we picked `"steamid"`,`"appid"`as user and item features, and treat `"voted-up"` as explicit rating.

To improve model perfermance, we decided to filter out cold starters and set thresholds:
1. Take off 'unpopular' games (less than 100 user rating)
2. Take off 'unfrequent' users (rated less than 10 games)

Also, we encoded steamid and appid with indexStringColumns to compress the data. And converted all data to integer to comply with the model setting.

```python
als = ALS(maxIter=5, regParam=0.01, userCol="steamid", 
          itemCol="appid", ratingCol="rating")
model = als.fit(df_training)
predictions = model.transform(df_test)
predictions = predictions.na.drop()
```

To evaluate the model perfermance, we picked RMSE with code below:
```python
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```
A useful predictive model normally come with 0.2~0.5 RMSE, and you can use this metric to check the model effectiveness.

We finally leveraged the model to predict top 10 recommendated games to each user in the validation data set:
```python
userRecs = model.recommendForAllUsers(10)
userRecs.show()
```
