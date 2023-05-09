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
    
    train = spark.read.parquet(f'hdfs:/user/xl4703_nyu_edu/ALS_train_100.parquet')
    validation = spark.read.parquet(f'hdfs:/user/xl4703_nyu_edu/ALS_validation_100.parquet')
    ranking = spark.read.parquet(f'hdfs:/user/xl4703_nyu_edu/user_norm_rank_100.parquet')

    # hyperparameter tuning
    regs = [0.1, 10, 100, 1000]
    rs = [50, 100, 150]
    alphas = [0.1, 10, 100, 1000]
    hyperparams = [regs, rs, alphas]
    hp_list = list(itertools.product(*hyperparams))
    count = 0
    
    best_ndcg = -100
    best_parameter_ndcg = []
    best_meanAP = -100
    best_parameter_MAP = []
    k = 100
    for hp in hp_list:
        count += 1
        print('total:', len(hp_list), 'currently at:', count)
        

        # Build the recommendation model using ALS on the training data
        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        als = ALS(maxIter=5, regParam=hp[0], rank = hp[1], alpha= hp[2],
                  userCol="user_id", itemCol="track_new_id", ratingCol="normalized_ranking",
                  coldStartStrategy="drop")
        model = als.fit(train)

        # meanAP and NCDG
        
        # Prepare the true labels
        true_labels = (
            validation
            .groupBy("user_id")
            .agg(collect_list("track_new_id").alias("true_tracks"))
            .select("user_id", "true_tracks")
        )

        # Get the top k recommendations for each user

        user_recs = model.recommendForAllUsers(k).select("user_id", "recommendations.track_new_id")

        # Join the true labels with the recommendations
        predictions_and_labels = (
            true_labels
            .join(user_recs, on="user_id")
            .select("track_new_id", "true_tracks")
        )

        # Convert the DataFrame to an RDD
        predictions_and_labels_rdd = predictions_and_labels.rdd.map(tuple)

        # Initialize RankingMetrics with the RDD of (predicted, true) label pairs
        metrics = RankingMetrics(predictions_and_labels_rdd)

        # Calculate Mean Average Precision (MAP)
        mean_ap = metrics.meanAveragePrecision
        #print("Mean Average Precision (MAP) = ", mean_ap)

        # Calculate Normalized Discounted Cumulative Gain (NDCG) at k
        ndcg_at_k = metrics.ndcgAt(k)
        #print(f"Normalized Discounted Cumulative Gain (NDCG) at {k} = ", ndcg_at_k)

        if ndcg_at_k > best_ndcg:
            best_ndcg = ndcg_at_k
            best_parameter_ndcg = hp
        if mean_ap > best_meanAP:
            best_meanAP = mean_ap
            best_parameter_MAP = hp
        print('current best parameter sets:', best_parameter_ndcg, 'with best ndcg@100:', best_ndcg)
        print('current best parameter sets:', best_parameter_MAP, 'with best meanAP@100:', best_meanAP)
            
    print('Parameter setting with best performance:', best_parameter_ndcg, 'with ndcg@100:', best_ndcg)
    print('Parameter setting with best performance:', best_parameter_MAP, 'with meanAP@100:', best_meanAP)


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




