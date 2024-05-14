from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, to_timestamp, desc

# import all necesary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import calendar
import plotly as pt
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from pylab import *
import matplotlib.patheffects as PathEffects

import descartes
import geopandas as gpd
from Levenshtein import distance
from itertools import product
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point, Polygon

import geoplot
from geopy.geocoders import Nominatim

import warnings

warnings.filterwarnings('ignore')

# Spark Session oluştur
spark = SparkSession.builder \
    .appName("Accident Analysis") \
    .getOrCreate()

# Veriyi yükle
df = spark.read.csv("US_Accidents_March23.csv", header=True)

# Kolon ve satır sayısını kontrol et
print("The Dataset Contains, Rows: {:,d} & Columns: {}".format(df.count(), len(df.columns)))

# Tarih saat sütunlarını dönüştür
df = df.withColumn("Start_Time", to_timestamp(col("Start_Time")))
df = df.withColumn("End_Time", to_timestamp(col("End_Time")))

# Şehirler ve kaza sayıları
city_df = df.groupBy("City").agg(count("*").alias("Cases")).orderBy(col("Cases").desc())

# En yüksek kaza sayısı ve hesaplamaları
highest_cases = city_df.select("Cases").first()[0]
print(round(highest_cases / 5))
print(round(highest_cases / (5 * 365)))

# Top 10 şehir
top_10_cities = city_df.limit(10)

# Sonuçları görüntüle
top_10_cities.show()

top_10_citiesPD = top_10_cities.toPandas()
city_dfPd = city_df.toPandas()

########################################################################################################################
"""
fig, ax = plt.subplots(figsize=(12, 7), dpi=80)

cmap = cm.get_cmap('rainbow', 10)
clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

ax = sns.barplot(y=top_10_citiesPD['Cases'], x=top_10_citiesPD['City'], palette='rainbow')

total = sum(city_dfPd['Cases'])
for i in ax.patches:
    ax.text(i.get_x() + .03, i.get_height() - 10500, \
            str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=15, weight='bold',
            color='white')

plt.title('\nTop 10 Cities in US with most no. of \nRoad Accident Cases (2016-2023)\n', size=20, color='grey')

plt.ylim(1000, 200000)
plt.xticks(rotation=10, fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel('\nCities\n', fontsize=15, color='grey')
ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

for i in ['bottom', 'left']:
    ax.spines[i].set_color('white')
    ax.spines[i].set_linewidth(1.5)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_axisbelow(True)
ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)
plt.show()
"""

########################################################################################################################

# create a dictionary using US State code and their corresponding Name
us_states = {'AK': 'Alaska',
             'AL': 'Alabama',
             'AR': 'Arkansas',
             'AS': 'American Samoa',
             'AZ': 'Arizona',
             'CA': 'California',
             'CO': 'Colorado',
             'CT': 'Connecticut',
             'DC': 'District of Columbia',
             'DE': 'Delaware',
             'FL': 'Florida',
             'GA': 'Georgia',
             'GU': 'Guam',
             'HI': 'Hawaii',
             'IA': 'Iowa',
             'ID': 'Idaho',
             'IL': 'Illinois',
             'IN': 'Indiana',
             'KS': 'Kansas',
             'KY': 'Kentucky',
             'LA': 'Louisiana',
             'MA': 'Massachusetts',
             'MD': 'Maryland',
             'ME': 'Maine',
             'MI': 'Michigan',
             'MN': 'Minnesota',
             'MO': 'Missouri',
             'MP': 'Northern Mariana Islands',
             'MS': 'Mississippi',
             'MT': 'Montana',
             'NC': 'North Carolina',
             'ND': 'North Dakota',
             'NE': 'Nebraska',
             'NH': 'New Hampshire',
             'NJ': 'New Jersey',
             'NM': 'New Mexico',
             'NV': 'Nevada',
             'NY': 'New York',
             'OH': 'Ohio',
             'OK': 'Oklahoma',
             'OR': 'Oregon',
             'PA': 'Pennsylvania',
             'PR': 'Puerto Rico',
             'RI': 'Rhode Island',
             'SC': 'South Carolina',
             'SD': 'South Dakota',
             'TN': 'Tennessee',
             'TX': 'Texas',
             'UT': 'Utah',
             'VA': 'Virginia',
             'VI': 'Virgin Islands',
             'VT': 'Vermont',
             'WA': 'Washington',
             'WI': 'Wisconsin',
             'WV': 'West Virginia',
             'WY': 'Wyoming'
             }

# Eyaletler ve karşılık gelen kaza sayılarından oluşan bir DataFrame oluşturma
state_df = df.groupBy("State").agg(count("*").alias("Cases")) \
    .orderBy(desc("Cases")) \
    .select(col("State"), col("Cases")) \
    .toPandas()


# Eyalet kodlarını isimlerine dönüştüren bir fonksiyon tanımlama
def convert(state_code):
    return us_states.get(state_code, state_code)


# Eyalet kodlarını isimlerine dönüştürme
state_df['State'] = state_df['State'].apply(convert)

# Top 10 eyalet ismini alın
top_ten_states_name = state_df.head(10)

print(top_ten_states_name)

########################################################################################################################

fig, ax = plt.subplots(figsize=(12, 6), dpi=80)

cmap = cm.get_cmap('winter', 10)
clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

ax = sns.barplot(y=state_df['Cases'].head(10), x=state_df['State'].head(10), palette='winter')
ax1 = ax.twinx()
sns.lineplot(data=state_df[:10], marker='o', x='State', y='Cases', color='white', alpha=.8)

total = df.count()
for i in ax.patches:
    ax.text(i.get_x() - 0.2, i.get_height() + 10000,
            ' {:,d}\n  ({}%) '.format(int(i.get_height()), round(100 * i.get_height() / total, 1)), fontsize=15,
            color='black')

ax.set(ylim=(-10000, 2000000))
ax1.set(ylim=(-100000, 1700000))

plt.title('\nTop 10 States with most no. of \nAccident cases in US (2016-2023)\n', size=20, color='grey')
ax1.axes.yaxis.set_visible(False)
ax.set_xlabel('\nStates\n', fontsize=15, color='grey')
ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

for i in ['top', 'right']:
    side1 = ax.spines[i]
    side1.set_visible(False)
    side2 = ax1.spines[i]
    side2.set_visible(False)

ax.set_axisbelow(True)
ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)

ax.spines['bottom'].set_bounds(0.005, 9)
ax.spines['left'].set_bounds(0, 600000)
ax1.spines['bottom'].set_bounds(0.005, 9)
ax1.spines['left'].set_bounds(0, 600000)
ax.tick_params(axis='y', which='major', labelsize=10.6)
ax.tick_params(axis='x', which='major', labelsize=10.6, rotation=10)

plt.show()

########################################################################################################################



# Spark Session'ı kapat
spark.stop()
