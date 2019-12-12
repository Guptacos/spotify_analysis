import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
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
                   'valence', 'tempo', 'duration_ms']

features = df[feature_cols].values
labels = df['date'].apply(getYearRange).values



def learningRateSearch(features, labels, n_splits, vals):
	accs = [0]*len(vals)
	#data_xgb = xgb.DMatrix(features, label=labels)
	for i, val in enumerate(vals):
		model = xgb.XGBClassifier(learning_rate=val)
		cross_val = xgb.cv(model.get_xgb_params(), features, num_boost_round=model.get_params()['n_estimators'], nfold=5,
	            metrics='merror', early_stopping_rounds=50)
		model.set_params(n_estimators=cvresult.shape[0])
		train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.2)
		model.fit(train_features, train_labels, eval_metric='merror')
		label_pred = model.predict(test_features)
		accs[i]+= metrics.accuracy_score(labels[test_idx], label_pred)

	"""accs = [0]*len(vals)
	for i, val in enumerate(vals):
		DT = xgb.XGBClassifier(n_estimators=val)
		DT = DT.fit(features[train_idx], labels[train_idx])
		label_pred = DT.predict(features[test_idx])
		accs[i]+= metrics.accuracy_score(labels[test_idx], label_pred)
	print(f"Split {j}: complete")"""

	for i in range(len(accs)):
		accs[i] = accs[i]/n_splits*100
		print(f"Accuracy for learning_rate={vals[i]}: {accs[i]}")

	return accs

def maxDepthSearch(features, labels, n_splits, vals):
	kf = KFold(n_splits)
	accs = [0]*len(vals)
	for j, (train_idx, test_idx) in enumerate(kf.split(features)):
		for i, val in enumerate(vals):
			DT = xgb.XGBClassifier(max_depth=val)
			DT = DT.fit(features[train_idx], labels[train_idx])
			label_pred = DT.predict(features[test_idx])
			accs[i]+= metrics.accuracy_score(labels[test_idx], label_pred)
		print(f"Split {j}: complete")

	for i in range(len(accs)):
		accs[i] = accs[i]/n_splits*100
		print(f"Accuracy for max_depth={vals[i]}: {accs[i]}")

	return accs

def individualSearch():
	"""depth_accs = maxDepthSearch(features, labels, 5, max_depth)
	plt.plot(max_depth, depth_accs)
	plt.xlabel("Max Depth Parameter")
	plt.ylabel("% Accuracy")
	plt.title("Accuracy vs. Max Tree Depth")
	plt.figure()"""


	learning_accs = learningRateSearch(features, labels, 5, [.1])
	"""plt.plot(num_estimators, impurity_accs)
	plt.xlabel("Number of Estimators Parameter")
	plt.ylabel("% Accuracy")
	plt.title("Accuracy vs. Number of Trees in the Forest ")
	plt.show()"""

max_depth = [5, 8, 10, 14, 20, 25, 30]
learning_rates = [.01, .05, .1, .2, .3, .5]

individualSearch()