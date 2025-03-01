# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

#instantiate the spark session
spark = SparkSession.builder.appName("Demo").getOrCreate()

# COMMAND ----------

# Part A1
integer = spark.sparkContext.textFile("/FileStore/tables/integer.txt")
integer.collect()

arrayRDD = integer.map(lambda x: int(x)%2)
arrayRDD.collect()

reduce = arrayRDD.map(lambda word: (word, 1))
reduce.collect()

res = reduce.reduceByKey(lambda x,y : x + y) 

print("Even Number: ", res.collect()[0][1])
print("Odd Number: ", res.collect()[1][1])    

# COMMAND ----------

# Part A2
salary = spark.sparkContext.textFile("/FileStore/tables/salary.txt")
salary.collect()

arrayRDD = salary.map(lambda x: x.split(" "))
arrayRDD.collect()

toInt = arrayRDD.map(lambda x : (x[0], int(x[1]))) 
toInt.collect()

sumRDD = toInt.reduceByKey(lambda x,y : x + y)
salary_df = spark.createDataFrame(toInt, ["Department", "Salary"])
stats_df = salary_df.groupBy("Department").agg(
    avg("Salary").alias("Mean"),
    expr("percentile_approx(Salary, 0.5)").alias("Median"),
    stddev("Salary").alias("Std")
)
stats_df.show()


# COMMAND ----------

# Convert to Pandas for visualization
salary_pd = salary_df.toPandas()

# Boxplot for Salary Distribution
plt.figure(figsize=(10,6))
sns.boxplot(x="Department", y="Salary", data=salary_pd)
plt.title("Salary Distribution by Department")
plt.xticks(rotation=45)
plt.show()

# Histogram of Salaries
plt.figure(figsize=(8,6))
sns.histplot(salary_pd["Salary"], bins=20)
plt.title("Overall Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# COMMAND ----------

# Part A3
shakespeare = spark.sparkContext.textFile("/FileStore/tables/shakespeare_1-1.txt")

words = shakespeare.flatMap(lambda line: line.split(" "))
wordCounts = words.map(lambda word : (word, 1)).reduceByKey(lambda x,y : x+y)

result = []
word_list = ["Shakespeare", "What", "The", "Lord", "Library", "GUTENBERG", "WILLIAM", "COLLEGE", "WORLD"]
for i in wordCounts.collect():
    if i[0] in word_list:
        result.append(i)
        
result

# COMMAND ----------

# Part A4
import re

def clean_text(line):
    line = re.sub(r"[^\w\s]", "", line)  # Remove punctuation
    words = line.lower().split()
    return words

# Apply text cleaning and word count
wordsRDD = shakespeare.flatMap(clean_text).map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)

top_10_words = wordsRDD.sortBy(lambda x: x[1], ascending=False).take(10)
bottom_10_words = wordsRDD.sortBy(lambda x: x[1], ascending=True).take(10)

# Display results
print("Top 10 Most Common Words")
for word, count in top_10_words:
    print(f"{word}: {count}")

# COMMAND ----------

print("Bottom 10 Least Common Words")
for word, count in bottom_10_words:
    print(f"{word}: {count}")

# COMMAND ----------

# Part B1
from pyspark.sql.functions import col, avg, count
# identify the top 10 movies with the highest average rating
movies = spark.sparkContext.textFile("/FileStore/tables/movies.csv")

header = movies.first()
movie_data = movies.filter(lambda row: row != header)
linedata = movie_data.map(lambda line: line.split(","))
spark_data = linedata.map(lambda fields: Row(movieId=int(fields[0]), rating=float(fields[1]), userId=int(fields[2])))
spark_df = spark.createDataFrame(spark_data)

# # Alternative Laoding method
# path = "/FileStore/tables/movies.csv"
# spark_df = spark.read \
#   .format("csv") \
#   .option("inferSchema", True) \
#   .option("header", True) \
#   .option("sep", ',') \
#   .option("path", path) \
#   .load()


top_rating = spark_df.groupBy("movieId").agg(
    avg("rating").alias('average rating'),
    count("rating").alias('rating times'),
)
top_rating = top_rating.orderBy(col("average rating"), ascending = False)
top_rating.show(10)

# COMMAND ----------

#  top 10 users who have contributed the most ratings
top_user = spark_df.groupBy('userId').agg(
    count("rating").alias("total rating time")
)
top_user = top_user.orderBy(col("total rating time"), ascending = False)
top_user.show(10)

# COMMAND ----------

# Barplot of rating distribution
user_ratings_tr = top_rating.toPandas()

plt.figure(figsize=(25,6))
sns.barplot(x=user_ratings_tr['movieId'], y=user_ratings_tr['average rating'])
plt.title("Distribution of Average Movie Ratings")
plt.xlabel("Movie ID")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Barplot of the number of ratings for each user
user_ratings = top_user.toPandas()

plt.figure(figsize=(12,6))
sns.barplot(x=user_ratings['userId'], y=user_ratings['total rating time'])
plt.title("Number of Ratings per User")
plt.xlabel("User ID")
plt.ylabel("Number of Ratings")
plt.show()


# COMMAND ----------

# Part B2
# Use randomSplit() if you want a quick, general split and are not worried about users missing from training.
# Use sampleBy() (Stratified Sampling) to ensure all users are represented proportionally in both training & test sets.

user_fractions_80_20 = {uid: 0.8 for uid in spark_df.select("userId").distinct().rdd.flatMap(lambda x: x).collect()}
user_fractions_70_30 = {uid: 0.7 for uid in spark_df.select("userId").distinct().rdd.flatMap(lambda x: x).collect()}

# Ratio1 80/20
train1 = spark_df.sampleBy("userId", fractions=user_fractions_80_20, seed=1)
test1 = spark_df.subtract(train1)

# Ratio2 70/30
train2 = spark_df.sampleBy("userId", fractions=user_fractions_70_30, seed=1)
test2 = spark_df.subtract(train2)

# Show sample sizes
print(f"Train1 (80%): {train1.count()}, Test1 (20%): {test1.count()}")
print(f"Train2 (70%): {train2.count()}, Test2 (30%): {test2.count()}")


# COMMAND ----------

# Part B2 continue
# Function to train and evaluate ALS Model
rmse_results = {}
def train_and_evaluate(train_df, test_df, split_name):
    als = ALS(
        maxIter=10, 
        regParam=0.1, 
        rank=10,
        userCol="userId", itemCol="movieId", ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(train_df)
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    rmse_results[split_name] = rmse
    print(f"Performance for {split_name} Split:")
    print(f"RMSE: {rmse:.4f}\n")

train_and_evaluate(train1, test1, "80/20")
train_and_evaluate(train2, test2, "70/30")

# COMMAND ----------

# Part B3
# Modify my function
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import collect_list
def train_and_evaluate_pro(train_df, test_df, split_name):
    als = ALS(
        maxIter=10, 
        regParam=0.1, 
        rank=10,
        userCol="userId", itemCol="movieId", ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(train_df)
    predictions = model.transform(test_df)

    # RMSE
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator_rmse.evaluate(predictions)

    # MAE
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    mae = evaluator_mae.evaluate(predictions)

    # MSE
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
    mse = evaluator_mse.evaluate(predictions)

    print(f"Performance for {split_name} Split:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

# COMMAND ----------

train_and_evaluate_pro(train1, test1, "80/20")
train_and_evaluate_pro(train2, test2, "70/30")

# COMMAND ----------

from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import collect_list, col

def evaluate_precision_recall_f1(train_df, test_df, split_name):
    als = ALS(
        maxIter=10, 
        regParam=0.1, 
        rank=10,
        userCol="userId", itemCol="movieId", ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(train_df)
    predictions = model.transform(test_df)

    # Convert movieId to Double for RankingEvaluator
    predictions = predictions.withColumn("movieId", col("movieId").cast("double"))
    # Convert predictions to ranking format (Top-N recommendations)
    k = 10
    top_k_predictions = predictions.groupBy("userId").agg(collect_list("movieId").alias("predicted_items"))
    # Create Ground Truth for RankingEvaluator
    actual_items = test_df.withColumn("movieId", col("movieId").cast("double")).groupBy("userId").agg(collect_list("movieId").alias("actual_items"))
    # Join predictions with ground truth
    ranking_data = top_k_predictions.join(actual_items, "userId")

    # Precision@K
    evaluator_precision = RankingEvaluator(metricName="precisionAtK", labelCol="actual_items", predictionCol="predicted_items", k=k)
    precision = evaluator_precision.evaluate(ranking_data)

    # Recall@K
    evaluator_recall = RankingEvaluator(metricName="recallAtK", labelCol="actual_items", predictionCol="predicted_items", k=k)
    recall = evaluator_recall.evaluate(ranking_data)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print Performance Results
    print(f"Performance for {split_name} Split:")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


# COMMAND ----------

evaluate_precision_recall_f1(train1, test1, "80/20")
evaluate_precision_recall_f1(train2, test2, "70/30")

# COMMAND ----------

# Part B4
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

als = ALS(
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating", 
    coldStartStrategy="drop"
)

paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(als.maxIter, [10, 15]) \
    .build()
    
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Perform 5-Fold Cross-Validation
crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cv_model = crossval.fit(train1)

best_model = cv_model.bestModel
best_rank = best_model.rank
best_regParam = best_model._java_obj.parent().getRegParam()
best_maxIter = best_model._java_obj.parent().getMaxIter()
print(f"Best Model Parameters: Rank = {best_rank}, RegParam = {best_regParam}, MaxIter = {best_maxIter}")

# COMMAND ----------

# Visualize the impact of different hyperparameter configurations on model RMSE scores
rmse_results = []

for i, params in enumerate(paramGrid):
    rmse = cv_model.avgMetrics[i]
    rank = [v for k, v in params.items() if "rank" in k.name][0] if any("rank" in k.name for k in params.keys()) else None
    regParam = [v for k, v in params.items() if "regParam" in k.name][0] if any("regParam" in k.name for k in params.keys()) else None
    maxIter = [v for k, v in params.items() if "maxIter" in k.name][0] if any("maxIter" in k.name for k in params.keys()) else None
    rmse_results.append((rank, regParam, maxIter, rmse))

rmse_df = pd.DataFrame(rmse_results, columns=["Rank", "RegParam", "MaxIter", "RMSE"])

# Visualize RMSE Results
plt.figure(figsize=(10,6))
for maxIter in [10, 15]:
    subset = rmse_df[rmse_df["MaxIter"] == maxIter]
    plt.plot(subset["Rank"].astype(str) + "-" + subset["RegParam"].astype(str), subset["RMSE"], 
             marker="o", label=f"MaxIter {maxIter}")
    
plt.xlabel("Rank - RegParam")
plt.ylabel("RMSE")
plt.title("RMSE for Different ALS Hyperparameter Combinations")
plt.legend()
plt.show()


# COMMAND ----------

# Part B5
# Using the hyperparameter tuned from the last question
als = ALS(
    maxIter=10, 
    regParam=0.1, 
    rank=10,
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Fit the model to the training data
model = als.fit(train1)

user_ids = [11, 21]
users_df = spark.createDataFrame([(uid,) for uid in user_ids], ["userId"])
user_recommendations = model.recommendForUserSubset(users_df, 5)
user_recommendations.show(truncate=False)

# COMMAND ----------

# Retrieve Actual Ratings Given by User 11 and 21
actual_ratings_11 = spark_df.filter(col("userId").isin(11))
top_5_actual_ratings_11 = actual_ratings_11.orderBy(col("rating").desc()).limit(5)
top_5_actual_ratings_11.show()


# COMMAND ----------

actual_ratings_21 = spark_df.filter(col("userId").isin(21))
top_5_actual_ratings_21 = actual_ratings_21.orderBy(col("rating").desc()).limit(5)
top_5_actual_ratings_21.show()
