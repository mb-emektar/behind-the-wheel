from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col, count, to_timestamp, desc, date_format, dayofmonth, year, round, stddev, \
    median as spark_median, lit
import pyspark.sql.functions as F

import warnings

warnings.filterwarnings('ignore')

# Spark Session oluştur
spark = SparkSession.builder \
    .appName("Predict Severity") \
    .getOrCreate()

df = spark.read.csv("US_Accidents_March23.csv", header=True, inferSchema=True)

# 'Start_Time' ve 'End_Time' sütunlarını timestamp türüne dönüştür
df = df.withColumn('Start_Time', to_timestamp(col('Start_Time'))) \
    .withColumn('End_Time', to_timestamp(col('End_Time')))

# Yıl, Ay, Gün, Saat ve Haftanın Günü sütunlarını çıkar
df = df.withColumn('Year', year(col('Start_Time'))) \
    .withColumn('Month', date_format(col('Start_Time'), 'MMM')) \
    .withColumn('Day', dayofmonth(col('Start_Time'))) \
    .withColumn('Hour', hour(col('Start_Time'))) \
    .withColumn('Weekday', date_format(col('Start_Time'), 'E'))

# Kaza süresini dakika cinsinden hesapla
df = df.withColumn('Time_Duration(min)', round((col('End_Time').cast("long") - col('Start_Time').cast("long")) / 60))

# Negatif veya sıfır süreleri içeren satırları filtrele ve sil
df = df.filter(col('Time_Duration(min)') > 0)

# Medyan ve standart sapma hesapla
td = 'Time_Duration(min)'
median_val = df.approxQuantile(td, [0.5], 0.01)[0]
stddev_val = df.select(stddev(col(td))).collect()[0][0]

# Outlier'ları NaN ile değiştir
df = df.withColumn(td, F.when(F.abs(col(td) - median_val) > stddev_val * 3, None).otherwise(col(td)))


# NaN değerleri medyan ile doldur
df = df.na.fill({td: median_val})

# Kaza süresi bilgilerini yazdır
max_td = df.agg({"Time_Duration(min)": "max"}).collect()[0][0]
min_td = df.agg({"Time_Duration(min)": "min"}).collect()[0][0]

print(max_td)
print(min_td)
df.printSchema()


# Makine öğrenmesi için kullanılacak özellikleri seç
feature_lst = ['Source','Severity','Start_Lng','Start_Lat','Distance(mi)','City','County','State','Timezone','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Hour','Weekday', 'Time_Duration(min)']

# Seçilen özellikleri içeren yeni bir DataFrame oluştur
df_sel = df.select(*feature_lst)

# Yeni DataFrame'in şemasını görüntüle
df_sel.printSchema()

# İlk 5 satırı görüntüle
df_sel.show(5)