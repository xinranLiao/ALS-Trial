#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import argparse

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


def main(spark, args, userID):
     
    train = spark.read.parquet(f'hdfs:/user/xl4703_nyu_edu/ALS_train_100.parquet')
    validation = spark.read.parquet(f'hdfs:/user/xl4703_nyu_edu/ALS_validation_100.parquet')
    ranking = spark.read.parquet(f'hdfs:/user/xl4703_nyu_edu/user_norm_rank_100.parquet')

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    reg = args.reg
    r = args.rank
    a = args.alpha
    
    als = ALS(maxIter=5,regParam=reg, rank = r, alpha=a, 
                          userCol="user_id", itemCol="track_new_id", ratingCol="normalized_ranking",
                          coldStartStrategy="drop")
    model = als.fit(train)

    # Prepare the true labels
    true_labels = (
        validation
        .groupBy("user_id")
        .agg(collect_list("track_new_id").alias("true_tracks"))
        .select("user_id", "true_tracks")
    )

    # Get the top k recommendations for each user
    k = 100
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
    print("Mean Average Precision (MAP) = ", mean_ap)

    # Calculate Normalized Discounted Cumulative Gain (NDCG) at k
    ndcg_at_k = metrics.ndcgAt(k)
    print(f"Normalized Discounted Cumulative Gain (NDCG) at {k} = ", ndcg_at_k)




        
        


# In[37]:


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-reg', '--reg', type=float, required=False, default =0.0, help='hyperparameter regterm')
    parser.add_argument('-alpha', '--alpha', type=float, required=False, default =0.0, help='hyperparameter alpha')
    parser.add_argument('-rank', '--rank', type=float, required=False, default =0.0, help='hyperparameter rank')
    args = parser.parse_args()
    
    sc = spark.sparkContext
    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, args, userID)


# In[ ]:





# In[ ]:




