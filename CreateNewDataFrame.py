from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col, count, to_timestamp, desc, date_format, dayofmonth, year, round, stddev, \
    median as spark_median, lit
import pyspark.sql.functions as F

import warnings

warnings.filterwarnings('ignore')

# Create Spark Session
spark = SparkSession.builder \
    .appName("Predict Severity") \
    .getOrCreate()

df = spark.read.csv("US_Accidents_March23.csv", header=True, inferSchema=True)

# Convert 'Start_Time' and 'End_Time' columns into timestamp
df = df.withColumn('Start_Time', to_timestamp(col('Start_Time'))) \
    .withColumn('End_Time', to_timestamp(col('End_Time')))

# Extract year month day hour weekday columns
df = df.withColumn('Year', year(col('Start_Time'))) \
    .withColumn('Month', date_format(col('Start_Time'), 'MMM')) \
    .withColumn('Day', dayofmonth(col('Start_Time'))) \
    .withColumn('Hour', hour(col('Start_Time'))) \
    .withColumn('Weekday', date_format(col('Start_Time'), 'E'))

# Calculate time duration as minutes
df = df.withColumn('Time_Duration(min)', round((col('End_Time').cast("long") - col('Start_Time').cast("long")) / 60))

# Delete rows with negative or 0 values of time duration
df = df.filter(col('Time_Duration(min)') > 0)

# Calculate median and standard deviation
td = 'Time_Duration(min)'
median_val = df.approxQuantile(td, [0.5], 0.01)[0]
stddev_val = df.select(stddev(col(td))).collect()[0][0]

# Change outliers with NaN
df = df.withColumn(td, F.when(F.abs(col(td) - median_val) > stddev_val * 3, None).otherwise(col(td)))

# Fill NaN values with median
df = df.na.fill({td: median_val})

# Choose feature columns for Machine Learning algorithms
feature_lst = ['Severity', 'Start_Lng', 'Start_Lat', 'Distance(mi)', 'City', 'County', 'State', 'Timezone',
               'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Weather_Condition',
               'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
               'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset', 'Hour', 'Weekday',
               'Time_Duration(min)']

# Create new dataframe with selected features.
df_sel = df.select(*feature_lst)

# Delete rows with null values
selected_columns = [col_name for col_name in df_sel.columns if
                    df_sel.select(col_name).na.drop().count() != df_sel.count()]
df_sel = df_sel.dropna(subset=selected_columns)

# Print dataframe row and column size
print((df_sel.count(), len(df_sel.columns)))

# Write dataframe into a csv file
df_sel.write.csv('US_Accidents_March23_clean_sel_dropna.csv', header=True, mode='overwrite')

# Stop spark session
spark.stop()
