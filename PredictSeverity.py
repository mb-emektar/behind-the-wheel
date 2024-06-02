from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import pandas, matplotlib.pyplot
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')

import warnings

warnings.filterwarnings('ignore')

# Create Spark Session
spark = SparkSession.builder \
    .appName("Predict Severity2") \
    .getOrCreate()

df_sel_dropna = spark.read.csv("US_Accidents_March23_clean_sel_dropna.csv", header=True, inferSchema=True)

# Choose the state
state = 'CA'

# Choose state as California
df_state = df_sel_dropna.filter(col('State') == state).drop('State')

# Set county
county = 'Alameda'

# Select the county of Alameda
df_county = df_state.filter(col('County') == county).drop('County')


df_county.show()


# Convert Apache Spark DataFrame into Pandas DataFrame
df_county_pd = df_county.toPandas()

# Generate dummies for categorical data
df_county_dummy_pd = pd.get_dummies(df_county_pd, drop_first=True)

# Assign the data
df = spark.createDataFrame(df_county_dummy_pd)

# Show dataframe info
df.show()

# Set the target for the prediction
target = 'Severity'

# Combine features as vector
feature_columns = [col for col in df.columns if col != target]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df)

# Split data into train and test data
(train_data, test_data) = df_assembled.randomSplit([0.8, 0.2], seed=21)

# Create evaluator
evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction")

########################################################################################################################
########################################################################################################################
# Logistic Regression Model

# Create and fit Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol=target)
lr_model = lr.fit(train_data)

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Calculate scores
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("[Logistic regression algorithm] accuracy_score: {:.3f}".format(accuracy))
print("[Logistic regression algorithm] precision_score: {:.3f}".format(precision))
print("[Logistic regression algorithm] recall_score: {:.3f}".format(recall))
print("[Logistic regression algorithm] f1_score: {:.3f}".format(f1))

########################################################################################################################
########################################################################################################################
# Decision Tree Model

# Create and fit Decision Tree Model
dt = DecisionTreeClassifier(labelCol="Severity", featuresCol="features")
dt_model = dt.fit(train_data)

# Make predictions on test data
dt_predictions = dt_model.transform(test_data)

# Calculate scores
dt_accuracy = evaluator.evaluate(dt_predictions, {evaluator.metricName: "accuracy"})
dt_precision = evaluator.evaluate(dt_predictions, {evaluator.metricName: "weightedPrecision"})
dt_recall = evaluator.evaluate(dt_predictions, {evaluator.metricName: "weightedRecall"})
dt_f1 = evaluator.evaluate(dt_predictions, {evaluator.metricName: "f1"})

print("[Decision Tree algorithm] accuracy_score: {:.3f}".format(dt_accuracy))
print("[Decision Tree algorithm] precision_score: {:.3f}".format(dt_precision))
print("[Decision Tree algorithm] recall_score: {:.3f}".format(dt_recall))
print("[Decision Tree algorithm] f1_score: {:.3f}".format(dt_f1))

# Get feature importances for decision tree
feature_importances = dt_model.featureImportances.toArray()

# Print feature importances for decision tree
feature_importance_list = sorted(zip(feature_columns, feature_importances), key=lambda x: x[1], reverse=True)[:10]
print("Feature Importances for decision tree:")
for feature, importance in feature_importance_list:
    print(f"{feature}: {importance}")

########################################################################################################################
########################################################################################################################
# Random Forest Model

# Create and fit Random Forest Model
rf = RandomForestClassifier(labelCol="Severity", featuresCol="features", numTrees=100)
rf_model = rf.fit(train_data)

# Make predictions on test data
rf_predictions = rf_model.transform(test_data)

# Calculate scores
rf_accuracy = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})
rf_precision = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedPrecision"})
rf_recall = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedRecall"})
rf_f1 = evaluator.evaluate(rf_predictions, {evaluator.metricName: "f1"})

print("[Random Forest algorithm] accuracy_score: {:.3f}".format(rf_accuracy))
print("[Random Forest algorithm] precision_score: {:.3f}".format(rf_precision))
print("[Random Forest algorithm] recall_score: {:.3f}".format(rf_recall))
print("[Random Forest algorithm] f1_score: {:.3f}".format(rf_f1))

# Get feature importances
feature_importances = rf_model.featureImportances.toArray()

# Print feature importances
feature_importance_list = sorted(zip(feature_columns, feature_importances), key=lambda x: x[1], reverse=True)[:10]
print("Feature Importances:")
for feature, importance in feature_importance_list:
    print(f"{feature}: {importance}")


spark.stop()
