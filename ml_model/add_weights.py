import pandas as pd
import pickle

with open('data/results/hot100.df', 'rb') as f:
	df = pickle.load(f)

features =['danceability', 'energy', 'key', 'loudness', 'mode',
				   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
				   'valence', 'tempo', 'duration_ms']

aggregation_f = {'date': 'min', 'artist': 'first', 'name':'first', 'position':'sum'}

for feature in features:
	aggregation_f[feature] = 'first'

def addWeights(df):
	df['position'] = df['position'].apply(lambda x: 101-x)
	df = df.groupby(['name', 'artist']).agg(aggregation_f)

	return df

df = addWeights(df)

with open('classifier_data', 'wb') as f:
	pickle.dump(df, f)


def getMostInfluentialSongs(df):
	print((df.sort_values('position', ascending=False)[['date','position']]).head(10))
#df = df.swaplevel()

getMostInfluentialSongs(df)