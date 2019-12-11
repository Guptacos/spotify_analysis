import pandas as pd
import pickle

features =['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo', 'duration_ms']

aggregation_f = {'date': 'first', 'artist': 'first', 'title':'first', 'position':'sum'}

for feature in features:
	aggregation_f[feature] = 'first'

def addWeights(df):
	df['position'] = df['position'].apply(lambda x: 101-x)
	df = df.groupby(['title', 'artist']).agg(aggregation_f)

	return df


labels = df['date'].values
features = df.drop('date').values

with open('labels', 'wb') as f:
	pickle.dump(labels, f)

with open('features', 'wb') as f:
	pickle.dump(features, f)