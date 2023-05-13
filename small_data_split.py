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
from pyspark.sql.functions import col, rand, row_number, when
from pyspark.sql.window import Window
import itertools

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

    # reassigning id to the tracks 1. for ALS parsing (only take numerical item id) 2. handle same item with different msid
    
    track_numeric_id = spark.sql("""
        Select *, dense_rank() over(ORDER BY unique_id) as track_new_id
        from
            (Select *, COALESCE(recording_mbid, recording_msid) as unique_id
            from tracks) T1
    """)
    # track_numeric_id.show()
    track_numeric_id.createOrReplaceTempView('track_numeric_id')

    interaction_new_id = spark.sql("""
                    select i.user_id, i.recording_msid, t.track_new_id
                    From train_interaction i
                    Left Join track_numeric_id t
                    On i.recording_msid = t.recording_msid
        """)
    #interaction_new_id.show()
    interaction_new_id.createOrReplaceTempView('interaction_new_id')

    user_rank = spark.sql("""
                   select user_id,track_new_id, count_pop/count_total as ranking
                   from
                   (SELECT distinct user_id,track_new_id,
                           count(track_new_id) over(partition by user_id) as count_total,
                           count(track_new_id) over(partition by user_id, track_new_id) as count_pop
                    from interaction_new_id) T
                    order by user_id asc, ranking desc

        """)
    #ser_rank.show()
    
    user_norm_rank = spark.sql("""
            SELECT user_id, track_new_id, ranking, normalized_ranking
            FROM (
            SELECT user_id, track_new_id, ranking,
                   ((ranking - min_ranking) / (max_ranking - min_ranking))*100 as normalized_ranking
            FROM (
                SELECT user_id, track_new_id, count_pop / count_total as ranking,
                       MIN(count_pop / count_total) OVER (PARTITION BY user_id) as min_ranking,
                       MAX(count_pop / count_total) OVER (PARTITION BY user_id) as max_ranking
                FROM (
                    SELECT DISTINCT user_id, track_new_id,
                           COUNT(track_new_id) OVER (PARTITION BY user_id) as count_total,
                           COUNT(track_new_id) OVER (PARTITION BY user_id, track_new_id) as count_pop
                    FROM interaction_new_id
                ) T1
            ) T2
        ) T3
        WHERE user_id IS NOT NULL AND track_new_id IS NOT NULL AND ranking IS NOT NULL AND normalized_ranking IS NOT NULL AND normalized_ranking > 0
        ORDER BY user_id ASC
        """)
    #user_norm_rank.show()
    
    #user_norm_rank.write.parquet(f'hdfs:/user/xl4703_nyu_edu/user_norm_rank_100.parquet')
    #print("user_norm_rank.parquet complete")
    
    
    
    
    #preprocess split
    df = user_norm_rank
    
    print("original size:")
    print(df.count())
    
    df = df.withColumn("random", rand())
    user_counts = df.groupBy("user_id").count().withColumnRenamed("count", "total_tracks")
    df = df.join(user_counts, on="user_id")
    df = df.withColumn("user_index", row_number().over(Window.partitionBy("user_id").orderBy("random")))
    df = df.withColumn("split_threshold", (col("total_tracks") * 0.5).cast("integer"))
    df = df.withColumn("dataset", when(col("user_index") <= col("split_threshold"), "train").otherwise("validation"))
    df = df.drop("total_tracks", "user_index", "split_threshold", "random")
    
   
    
    train = df.filter(col("dataset") == "train").drop("dataset")
    #validation = df.filter(col("dataset") == "validation").drop("dataset")
    print("train size:")
    print(train.count())
    
    train.write.parquet(f'hdfs:/user/xl4703_nyu_edu/ALS_train_50.parquet')
    print("ALS_train.parquet complete")
    
    #validation.write.parquet(f'hdfs:/user/xl4703_nyu_edu/ALS_validation_without0.parquet')
    #print("ALS_validation.parquet complete")

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




