from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col, count, to_timestamp, desc, date_format, dayofmonth, year, round, stddev, \
    median as spark_median, lit
import pyspark.sql.functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Import numpy, pandas, matpltlib.pyplot, sklearn modules and seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')

# Spark Session oluştur
spark = SparkSession.builder \
    .appName("Predict Severity2") \
    .getOrCreate()

df_sel_dropna = spark.read.csv("US_Accidents_March23_clean_sel_dropna.csv", header=True, inferSchema=True)

# Eyaleti belirle
state = 'TX'

# Pennsylvania eyaletini seç
df_state = df_sel_dropna.filter(col('State') == state).drop('State')

# Set county
county = 'Loving'

# Select the county of Pennsylvania
df_county = df_state.filter(col('County') == county).drop('County')

# Apache Spark DataFrame'ini Pandas DataFrame'ine dönüştür
df_county_pd = df_county.toPandas()

# Generate dummies for categorical dataCity_King Of Prussia
df_county_dummy_pd = pd.get_dummies(df_county_pd, drop_first=True)

# Bilgiyi göster
df_county_dummy_pd.info()

# Assign the data
df = spark.createDataFrame(df_county_dummy_pd)

# Set the target for the prediction
target = 'Severity'

# Özellikleri bir vektör olarak birleştirme
feature_columns = [col for col in df.columns if col != target]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df)

# Veri kümesini eğitim ve test olarak bölme
(train_data, test_data) = df_assembled.randomSplit([0.8, 0.2], seed=21)

# Logistic Regression modelini oluşturma ve eğitme
lr = LogisticRegression(featuresCol="features", labelCol=target)
lr_model = lr.fit(train_data)

# Test verisi üzerinde tahmin yapma
predictions = lr_model.transform(test_data)

# Doğruluk puanını hesaplama
evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("[Logistic regression algorithm] accuracy_score: {:.3f}".format(accuracy))
spark.stop()
