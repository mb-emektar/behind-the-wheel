from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col, count, to_timestamp, desc

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
df = spark.read.csv("US_Accidents_March23.csv", header=True, inferSchema=True)

# Kolon ve satır sayısını kontrol et
print("The Dataset Contains, Rows: {:,d} & Columns: {}".format(df.count(), len(df.columns)))

# Tarih saat sütunlarını dönüştür
df = df.withColumn("Start_Time", to_timestamp(col("Start_Time")))
df = df.withColumn("End_Time", to_timestamp(col("End_Time")))


########################################################################################################################
# City Analysis
'''
# Cities and number of accidents
city_df = df.groupBy("City").agg(count("*").alias("Cases")).orderBy(col("Cases").desc())

# Highest number of accidents and calculations
highest_cases = city_df.select("Cases").first()[0]
print(round(highest_cases / 5))
print(round(highest_cases / (5 * 365)))

# Top 10 cities
top_10_cities = city_df.limit(10)

# show reslults
top_10_cities.show()

top_10_citiesPD = top_10_cities.toPandas()
city_dfPd = city_df.toPandas()

fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

cmap = cm.get_cmap('rainbow', 10)
clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

ax = sns.barplot(y=top_10_citiesPD['Cases'], x=top_10_citiesPD['City'], palette='plasma')

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
ax.set_ylabel('\nNumber of Accident Cases\n', fontsize=15, color='grey')

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
'''

########################################################################################################################
# Hour Analysis
'''
# Extract time information from the Start_Time column
df_hours = df.withColumn("Hours", hour(col("Start_Time")))

# Calculating the number of accidents per hour
hour_df = df_hours.groupBy("Hours").agg(count("*").alias("Cases")).orderBy("Hours")
hour_df_pd = hour_df.toPandas()

fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

clrs = []
for x in hour_df_pd['Cases']:
    if int(hour_df_pd[hour_df_pd['Cases'] == x]['Hours']) <= 11:
        if (x == max(list(hour_df_pd['Cases'])[:12])):
            clrs.append('red')
        else:
            clrs.append('#c3b632')
    else:
        if (x == max(list(hour_df_pd['Cases'])[12:])):
            clrs.append('red')
        else:
            clrs.append('#60154a')
ax = sns.barplot(y=hour_df_pd['Cases'], x=hour_df_pd['Hours'], palette=clrs)
ax1 = ax.twinx()

sns.lineplot(data=hour_df_pd, marker='o', x='Hours', y='Cases', color='white', alpha=1)

total = df.count()
for i in ax.patches:
    ax.text(i.get_x(), i.get_height() + 1000, \
            str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=10,
            color='black')

plt.ylim(1000, 150000)
plt.title('\nRoad Accident Percentage \nfor different hours along the day\n', size=20, color='grey')

ax1.axes.yaxis.set_visible(False)
ax.set_xlabel('\nHours\n', fontsize=15, color='grey')
ax.set_ylabel('\nNumber of Accident Cases\n', fontsize=15, color='grey')

for i in ['bottom', 'top', 'left', 'right']:
    ax.spines[i].set_color('white')
    ax.spines[i].set_linewidth(1.5)
    ax1.spines[i].set_color('white')
    ax1.spines[i].set_linewidth(1.5)

ax.set_axisbelow(True)
ax.grid(color='#b2d6c7', linewidth=1, alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=12)

MA = mpatches.Patch(color='red', label='Hours with maximum\nnumber of accidents')
MO = mpatches.Patch(color='#c3b632', label='A.M.')
NI = mpatches.Patch(color='#60154a', label='P.M.')

ax.legend(handles=[MA, MO, NI], prop={'size': 10.5}, loc='upper left', borderpad=1, edgecolor='white');

plt.show()
'''
########################################################################################################################
'''  
# usa map severity graph made without spark

df = pd.read_csv('./US_Accidents_March23.csv')

print('The Dataset Contains, Rows: {:,d} & Columns: {}'.format(df.shape[0], df.shape[1]))

states = gpd.read_file('./us-states-map')
df.Start_Time = pd.to_datetime(df.Start_Time, format='mixed')
df.End_Time = pd.to_datetime(df.End_Time, format='mixed')

geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), dpi=150)
ax = [ax1,ax2,ax3,ax4]
#fig, ax = plt.subplots(figsize=

index = 0
severity_levels = ['Severity 1','Severity 2','Severity 3','Severity 4']
severity_colors = ['#00FF00','#0640FF','#5705FF','#FF001E']

for i in ax:

    i.set_xlim([-125,-65])
    i.set_ylim([22,55])

    states.boundary.plot(ax=i, color='black')

    geo_df[geo_df['Severity'] == (index+1)].plot(ax=i, markersize=1, color=severity_colors[index], marker='.', label=severity_levels[index])

    for x in ['bottom', 'top', 'left', 'right']:
        side = i.spines[x]
        side.set_visible(False)

    title = '\n{} Visualization in US Map'.format(severity_levels[index])

    i.set_title(title, fontsize=18, color='grey')

    index += 1


fig.show()
'''

########################################################################################################################

# State Analysis
'''
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

# Create a DataFrame with states and the corresponding number of accidents
state_df = df.groupBy("State").agg(count("*").alias("Cases")) \
    .orderBy(desc("Cases")) \
    .select(col("State"), col("Cases")) \
    .toPandas()


# Define a function that converts state codes to their names
def convert(state_code):
    return us_states.get(state_code, state_code)


# Convert state codes to their names
state_df['State'] = state_df['State'].apply(convert)

# Get the top 10 state names
top_ten_states_name = state_df.head(10)

print(top_ten_states_name)

fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

cmap = cm.get_cmap('winter', 10)
clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

ax = sns.barplot(y=state_df['Cases'].head(10), x=state_df['State'].head(10), palette='viridis')
ax1 = ax.twinx()

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
'''
########################################################################################################################

# Severity Analysis
''' 
severity_df = df.groupBy("Severity").agg(count("*").alias("Cases")).orderBy(col("Cases").desc())
# Transformation to Pandas
severity_df_pd = severity_df.toPandas()

fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

# Calculate the percentage of total cases
total_cases = severity_df_pd['Cases'].sum()
severity_df_pd['Percentage'] = (severity_df_pd['Cases'] / total_cases) * 100
clrs = ['#b4e6ee', '#14a3ee', '#fdf4b8', '#ff4f4e']

ax = sns.barplot(x=severity_df_pd['Severity'], y=severity_df_pd['Percentage'], palette=clrs)

plt.title('\nImpact on the Traffic due to the Accidents\n', size=20, color='grey')

plt.xticks(rotation=10, fontsize=12)
plt.yticks(fontsize=12)

# Custom tick labels
custom_labels = ['Severity-1', 'Severity-2', 'Severity-3', 'Severity-4']
ax.set_xticklabels(custom_labels, rotation=10, fontsize=12)

plt.ylim(0, 100)
ax.set_xlabel('\nSeverity\n', fontsize=15, color='grey')
ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

ax.set_axisbelow(True)
ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)

# Add percentages on top of each bar
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.2f}%', (x + width / 2, y + height * 1.02), ha='center')

fig.show()
'''
########################################################################################################################

# Road Condition Analysis
'''
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(12, 20))

road_conditions = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'Stop', 'No_Exit', 'Traffic_Signal', 'Turning_Loop']
colors = [('#004B95', '#519DE9'), ('#38812F', '#7CC674'), ('#005F60', '#73C5C5'), ('#7D1007', '#C9190B'),
          ('#C58C00', '#F4C145'),
          ('#C46100', '#EF9234'), ('#3C3D99', '#8481DD'), ('#6A6E73', '#B8BBBE')]
index = 0


def func(pct, allvals):
    absolute = int(round(pct / 100 * np.sum(allvals), 2))
    return "{:.2f}%\n({:,d} Cases)".format(pct, absolute)


for i in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:

    road_conditions_count = df.groupBy(road_conditions[index]).agg(count("*").alias("Count"))
    size = list(road_conditions_count.toPandas()["Count"])

    if len(size) != 2:
        size.append(0)

    labels = ['True', 'False']

    i.pie(size, labels=labels, colors=colors[index],
          autopct=lambda pct: func(pct, size), labeldistance=1.1,
          textprops={'fontsize': 12}, explode=[0, 0.2])

    title = '\nPresence of {}'.format(road_conditions[index])

    i.set_title(title, fontsize=18, color='grey')

    index += 1
fig.show()
'''
# Spark Session'ı kapat
spark.stop()
