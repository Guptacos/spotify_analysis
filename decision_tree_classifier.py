from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle



with open('classifier_data', 'rb') as f:
	df = pickle.load(f)

yearsPerClass = 10

def getYearRange(date):
	year = date[:4]
	beginYear = ((int(year)-1970)//yearsPerClass * yearsPerClass) + 1970
	return f"{beginYear}-{beginYear+yearsPerClass-1}"


feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo', 'duration_ms', 'position']

features = df[feature_cols]
labels = df['date'].apply(getYearRange).values

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.2)

train_weights = train_features['position'].values
train_features = train_features.drop('position', axis=1).values

test_features = test_features.drop('position', axis=1).values


decades = ['70s', '80s', '90s', '00s', '10s']
#print("With Weights:")
#decision tree
model = DecisionTreeClassifier()
model = model.fit(train_features, train_labels, train_weights)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))
print(list(zip(feature_cols[:-1],model.feature_importances_)))
disp = metrics.plot_confusion_matrix(model, test_features, test_labels, display_labels=decades, normalize='true')
disp.ax_.set_title('Decision Tree')

#random forest
model = RandomForestClassifier()
model = model.fit(train_features, train_labels, train_weights)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))
print(list(zip(feature_cols[:-1],model.feature_importances_)))
disp = metrics.plot_confusion_matrix(model, test_features, test_labels, display_labels=decades, normalize='true')
disp.ax_.set_title('Random Forest')

#gb decision tree
model = xgb.XGBClassifier()
model = model.fit(train_features, train_labels, train_weights)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))
print(list(zip(feature_cols[:-1],model.feature_importances_)))
disp = metrics.plot_confusion_matrix(model, test_features, test_labels, display_labels=decades, normalize='true')
disp.ax_.set_title('Gradient Boosted Tree')
plt.show()


"""print("\nWithout Weights:")
model = DecisionTreeClassifier()
model = model.fit(train_features, train_labels)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))

model = RandomForestClassifier()
model = model.fit(train_features, train_labels)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))

model = xgb.XGBClassifier()
model = model.fit(train_features, train_labels)
label_pred = model.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))"""