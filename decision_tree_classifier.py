from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

yearsPerClass = 5

def getYearRange(date):
	year = date[:4]
	beginYear = ((int(year)-1970)//yearsPerClass * yearsPerClass) + 1970
	return f"{beginYear}-{beginYear+yearsPerClass-1}"


feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo', 'duration_ms']

features = df[feature_cols]
labels = df[date].apply(getYearRange)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.2)

model = DecisionTreeClassifier()
model = model.fit(train_features, train_labels)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))