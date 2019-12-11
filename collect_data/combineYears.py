import pickle
import os
from os.path import isfile, join

def getHandles(f):
	return open(f, 'rb')

# Change this value to the name you saved the pickled results to without "Chart" and the year
# e.g. if I saved the output to "countryChart2019", I would set this to "country"
chart = 'hot100'
files = map(getHandles, [f for f in os.listdir(os.getcwd()) if (isfile(join(os.getcwd(), f)) and chart in f)])

allCharts = []
for f in files:
	L = pickle.load(f)
	allCharts.extend(L)
	f.close()

with open(chart+"Results", 'wb') as f:
	pickle.dump(allCharts, f)

print("Pickled")