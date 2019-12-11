import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np

features =['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo', 'duration_ms']

aggregation_f = {'date': 'first'}

for feature in features:
	aggregation_f[feature] = np.mean

weeksPerPoint = 156

"""result = {"date": []}
for feature in features:
	result[feature] = []
for j, date in enumerate(["2000-11-11", "2000-12-11", "2001-01-11", "2001-02-11"]):
	for i in range(10):
		result["date"].append(date)
		for k, feature in enumerate(features):
			result[feature].append(random.random())"""

df = pickle.load(open('../data/results/hot100.df', 'rb')).reset_index()
df['date'] = pd.to_datetime(df['date'])
grouped = df.groupby('date')[features].agg(np.mean).reset_index()

sectionNum = []
if weeksPerPoint > 1:
	num_sections = len(grouped) // weeksPerPoint
	for i in range(num_sections):
		sectionNum.extend([i]*weeksPerPoint)
	sectionNum.extend([num_sections]*(len(grouped) - num_sections*weeksPerPoint))
	grouped["section"] = sectionNum
	grouped = grouped.groupby('section').agg(aggregation_f).set_index('date')
else:
	grouped = grouped.set_index('date')

test = ['loudness']
labels1 = ['danceability','energy','mode','valence']
labels2 = ['speechiness','acousticness','liveness','instrumentalness']
ax = plt.subplot(111)
ax.set_xlabel('Date')
ax.set_ylabel('Spotify Feature Value')
for feature in test:
	ax.plot(grouped[feature],'o-', label=feature)
		
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
plt.tight_layout()
plt.show()

"""ax = plt.subplot(212)
ax.set_xlabel('Date')
ax.set_ylabel('Spotify Feature Value')
for feature in labels2:
	ax.plot(grouped[feature],'o-', label=feature)
#ax.plot(grouped['duration_ms'], 'o-')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
plt.tight_layout()
plt.show()
charts = {
	pop: pickle.load(open('pop-songs','rb')),
}

for chart in charts.keys():
	grouped = charts[chart].groupby('date')[features].agg(np.mean)"""
	

