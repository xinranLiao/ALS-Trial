#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col, round
import pandas as pd
import numpy as np
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import avg, count, col, round, collect_list, desc, row_number
from pyspark.sql import Window as w
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    tracks = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet')
    #users = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_train_small.parquet')
    interactions = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    #train_interaction = spark.read.parquet(f'hdfs:/user/yz4315_nyu_edu/interactions_trainV2_small.parquet')
    # Give the dataframe a temporary view so we can run SQL queries
    #tracks.createOrReplaceTempView('tracks')
    #users.createOrReplaceTempView('users')
    #interactions.createOrReplaceTempView('interactions')
    #train_interaction.createOrReplaceTempView('train_interaction')
    
    # train validation split and write them to parquet
    #train_interaction, validation_interaction = interactions.randomSplit([0.8, 0.2], seed=52)
    #train_interaction.write.parquet(f'hdfs:/user/yz4315_nyu_edu/interactions_trainV2_small.parquet')
    #validation_interaction.write.parquet(f'hdfs:/user/yz4315_nyu_edu/interactions_validation_small.parquet')
    #train_interaction = spark.read.csv('data_test.csv', schema='user_id INT, recording_msid STRING')
    interactions.createOrReplaceTempView('train_interaction')
    tracks.createOrReplaceTempView('tracks')
    user_rank = spark.sql("""
                   select user_id,recording_msid, count_pop/count_total as ranking
                   from
                   (SELECT distinct user_id,recording_msid,
                           count(recording_msid) over(partition by user_id) as count_total,
                           count(recording_msid) over(partition by user_id, recording_msid) as count_pop
                    from train_interaction) T
                    order by user_id asc, ranking desc

        """)
    #ser_rank.show()
    
    user_norm_rank = spark.sql("""
            SELECT user_id, recording_msid, ranking, normalized_ranking
            FROM (
            SELECT user_id, recording_msid, ranking,
                   (ranking - min_ranking) / (max_ranking - min_ranking) as normalized_ranking
            FROM (
                SELECT user_id, recording_msid, count_pop / count_total as ranking,
                       MIN(count_pop / count_total) OVER (PARTITION BY user_id) as min_ranking,
                       MAX(count_pop / count_total) OVER (PARTITION BY user_id) as max_ranking
                FROM (
                    SELECT DISTINCT user_id, recording_msid,
                           COUNT(recording_msid) OVER (PARTITION BY user_id) as count_total,
                           COUNT(recording_msid) OVER (PARTITION BY user_id, recording_msid) as count_pop
                    FROM train_interaction
                ) T1
            ) T2
        ) T3
        ORDER BY user_id ASC, normalized_ranking DESC
        """)
    #user_norm_rank.show()
    
    '''
    
    #ratingsRDD = user_norm_rank.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
    #                                     rating=float(p[2]), timestamp=long(p[3])))
    #ratings = spark.createDataFrame(user_norm_rank)
    (training, test) = user_norm_rank.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="recording_msid", ratingCol="normalized_ranking",
              coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)

    # Generate top 10 movie recommendations for a specified set of users
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of movies
    movies = ratings.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)

'''

    track_numeric_id = spark.sql("""
        Select *, COALESCE(recording_mbid, recording_msid)
        from tracks
    



    """) 
    
    track_numeric_id.show()
    


# In[37]:


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    sc = spark.sparkContext
    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)


# In[ ]:





# In[ ]:




