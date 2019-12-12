from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def maxDepthSearch(features, labels, n_splits, vals):
	kf = KFold(n_splits)
	accs = [0]*len(vals)
	for j, (train_idx, test_idx) in enumerate(kf.split(features)):
		for i, val in enumerate(vals):
			DT = RandomForestClassifier(max_depth=val)
			DT = DT.fit(features[train_idx], labels[train_idx])
			label_pred = DT.predict(features[test_idx])
			accs[i]+= metrics.accuracy_score(labels[test_idx], label_pred)
		print(f"Split {j}: complete")

	for i in range(len(accs)):
		accs[i] = accs[i]/n_splits*100
		print(f"Accuracy for max_depth={vals[i]}: {accs[i]}")

	return accs

def numEstimatorsSearch(features, labels, n_splits, vals):
	kf = KFold(n_splits)
	accs = [0]*len(vals)
	for j, (train_idx, test_idx) in enumerate(kf.split(features)):
		for i, val in enumerate(vals):
			DT = RandomForestClassifier(n_estimators=val)
			DT = DT.fit(features[train_idx], labels[train_idx])
			label_pred = DT.predict(features[test_idx])
			accs[i]+= metrics.accuracy_score(labels[test_idx], label_pred)
		print(f"Split {j}: complete")

	for i in range(len(accs)):
		accs[i] = accs[i]/n_splits*100
		print(f"Accuracy for n_estimators={vals[i]}: {accs[i]}")

	return accs

def individualSearch():
	depth_accs = maxDepthSearch(features, labels, 5, max_depth)
	plt.plot(max_depth, depth_accs)
	plt.xlabel("Max Depth Parameter")
	plt.ylabel("% Accuracy")
	plt.title("Accuracy vs. Max Tree Depth")
	plt.figure()


	impurity_accs = numEstimatorsSearch(features, labels, 5, num_estimators)
	plt.plot(num_estimators, impurity_accs)
	plt.xlabel("Number of Estimators Parameter")
	plt.ylabel("% Accuracy")
	plt.title("Accuracy vs. Number of Trees in the Forest ")
	plt.show()

#max_depth = [3, 5, 8, 12, 16, 20, 25, 30]
#num_estimators = [20, 40, 60, 80, 100, 120, 140,160, 180, 200]
#individualSearch()


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.2)
max_depth = [8, 12, 16, 20]
num_estimators = [60, 100, 140, 200]

grid = {
	'max_depth': max_depth,
	'n_estimators': num_estimators
}

RF = RandomForestClassifier()
clf = GridSearchCV(RF, grid, n_jobs=-1, verbose=1)
clf.fit(train_features, train_labels)
print(clf.best_params_)
label_pred = clf.predict(test_features)
print(metrics.accuracy_score(test_labels, label_pred))


"""def gridSearch(features, labels, n_splits, min_impurity_dec, max_feats):
	kf = KFold(n_splits)
	accs = [[0 for i in range(len(max_feats))] for j in range(len(min_impurity_dec))]
	for i, (train_idx, test_idx) in enumerate(kf.split(features)):
		for j, dec in enumerate(min_impurity_dec):
			for k, feat in enumerate(max_feats):
				DT = DecisionTreeClassifier(max_depth=8, min_impurity_decrease=dec, max_features=feat)
				DT = DT.fit(features[train_idx], labels[train_idx])
				label_pred = DT.predict(features[test_idx])
				accs[j][k]+= metrics.accuracy_score(labels[test_idx], label_pred)
		print(f"Split {i}: complete")

	for i in range(len(min_impurity_dec)):
		for j in range(len(max_feats)):
			accs[i][j] = accs[i][j]/n_splits*100
			print(f"Accuracy for ({min_impurity_dec[i]},{max_feats[j]}): {accs[i][j]}")

	return accs

accs = gridSearch(features, labels, 5, min_impurity_dec, feats)
feats[0] = 'none'
indices = list(map(str, min_impurity_dec))
colors = ['b', 'r', 'g']
for x, fN, fS, fL in zip(indices, [accs[i][0] for i in range(len(accs))], [accs[i][1] for i in range(len(accs))], [accs[i][2] for i in range(len(accs))]):
	for i, (h, c) in enumerate(sorted(zip([fN, fS, fL], colors))):
		plt.bar(x, h,color=c, alpha=1, zorder=-i)

plt.legend(labels)
plt.legend(handles=[mpatches.Patch(color='b', label='Max Features=None'), mpatches.Patch(color='r', label='Max Features=Sqrt'), mpatches.Patch(color='g', label='Max Features=Log2')])
plt.xlabel("Min Impurity Decrease Parameter")
plt.ylabel("% Accuracy")
plt.show()"""