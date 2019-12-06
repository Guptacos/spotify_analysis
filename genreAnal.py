from radar import radar_factory
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
from numpy import random
import numpy as np



charts = {
	"alternative": pickle.load(open('data/results/alternative.df','rb')),
	"christian": pickle.load(open('data/results/christian.df','rb')),
	"country": pickle.load(open('data/results/country.df','rb')),
	"edm": pickle.load(open('data/results/edm.df','rb')),
	"jazz": pickle.load(open('data/results/jazz.df','rb')),
	"pop": pickle.load(open('data/results/pop.df','rb')),
	"rnbHipHop": pickle.load(open('data/results/rnbHipHop.df','rb')),
	"rock": pickle.load(open('data/results/rock.df','rb')),
}

features =['danceability', 'energy','mode',
				   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
				   'valence']
aggregation_f = {'date': 'first'}

for feature in features:
	aggregation_f[feature] = np.mean

weeksPerPoint = 52

for chart in charts.keys():
	df = charts[chart].reset_index()
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

	charts[chart] = grouped

"""for feature in charts["alternative"]:
	print(feature)
	print(charts["alternative"][feature].mean())
for feature in features:
	fig, ax = plt.subplots()
	ax.set_xlabel('Date')
	ax.set_ylabel('Spotify Feature Value')
	for chart in charts.keys():
		ax.plot(charts[chart][feature],'o-', label=chart)
			

	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	plt.tight_layout()
	plt.show()"""

"""N = len(features)
theta = radar_factory(N, frame='polygon')
spoke_labels = features
fig = plt.figure()
ax = fig.add_subplot(projection='radar')
colors = ['b', 'r', 'g', 'm', 'y', 'c', 'lime']
# Plot the four cases from the example data on separate axes
ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
ax.set_title("Genres", weight='bold', size='medium', position=(0.5, 1.1),
				horizontalalignment='center', verticalalignment='center')
ax.set_varlabels(spoke_labels)
labels = sorted(charts.keys())
legend = ax.legend(labels, loc=(0.9, .95),
				   labelspacing=0.1, fontsize='small')

def init():
	return step(0)


def step(i):
	data = []
	for chart in sorted(charts.keys()):
		genre = []
		for feature in features:
			genre.append(charts[chart][feature].loc['%d-01-01' % (2000+i):'%d-01-01' % (2000+i+1)].mean())
		data.append(genre)
	for d, color in zip(data, colors):
		if isinstance(genre[0], float):
			g, = ax.plot(theta, d, color=color)
			ax.fill(theta, d, facecolor=color, alpha=0.25)

	return g,




anim = animation.FuncAnimation(fig, step, init_func=init, frames=20, interval=1000, blit=True)

plt.show()"""



interval = 20

for i in range(2000, 2020, interval):
	N = len(features)
	theta = radar_factory(N, frame='polygon')
	data = []
	for chart in sorted(charts.keys()):
		genre = []
		for feature in features:
			v =charts[chart][feature].loc['%d-01-01' % i:'%d-01-01' % (i+interval)].mean()
			genre.append(v)
		data.append(genre)
	spoke_labels = features
	fig = plt.figure()
	ax = fig.add_subplot(projection='radar')
	#fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

	colors = ['b', 'r', 'g', 'm', 'y', 'c', 'lime']
	# Plot the four cases from the example data on separate axes
	ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
	ax.set_title("Genres", weight='bold', size='medium', position=(0.5, 1.1),
				 horizontalalignment='center', verticalalignment='center')
	for d, color in zip(data, colors):
		ax.plot(theta, d, color=color)
		ax.fill(theta, d, facecolor=color, alpha=0.25)
	ax.set_varlabels(spoke_labels)

	# add legend relative to top-left plot
	#ax = axes[0, 0]
	labels = sorted(charts.keys())
	plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
	plt.tight_layout()

	plt.show()

"""N = len(features)
theta = radar_factory(N, frame='polygon')
spoke_labels = features
fig = plt.figure()
ax = fig.add_subplot(projection='radar')
colors = ['b', 'r', 'g', 'm', 'y', 'c', 'lime']
# Plot the four cases from the example data on separate axes
ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
ax.set_title("Genres", weight='bold', size='medium', position=(0.5, 1.1),
				horizontalalignment='center', verticalalignment='center')
ax.set_varlabels(spoke_labels)
labels = sorted(charts.keys())
legend = ax.legend(labels, loc=(0.9, .95),
				   labelspacing=0.1, fontsize='small')


N = len(features)
theta = radar_factory(N, frame='polygon')
colors = ['b', 'r', 'g', 'm', 'y', 'c', 'lime']
#ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
#ax.set_varlabels(spoke_labels)
labels = sorted(charts.keys())
#legend = ax.legend(labels, loc=(0.9, .95),
#                   labelspacing=0.1, fontsize='small')

fig = plt.figure()
ax = fig.add_subplot()
line, = ax.plot([], [])

ax.set_title("Genres", weight='bold', size='medium', position=(0.5, 1.1),
				horizontalalignment='center', verticalalignment='center')

plotcols = ["black","red"]
lines = []
for index in range(2):
	lobj = ax.plot([],[],lw=2,color=colors[index])[0]
	lines.append(lobj)


def init():
	for line in lines:
		line.set_data([],[])
	return lines

# fake data
frame_num = 100
gps_data = [(1 * random.rand(2, frame_num)), (1 * random.rand(2, frame_num))]


def animate(i):
	x1,y1 = [],[]
	x2,y2 = [],[]
	x = gps_data[0][0, i]
	y = gps_data[1][0, i]
	x1.append(x)
	y1.append(y)

	x = gps_data[0][1,i]
	y = gps_data[1][1,i]
	x2.append(x)
	y2.append(y)

	xlist = [x1, x2]
	ylist = [y1, y2]

	#for index in range(0,1):
	for lnum,line in enumerate(lines):
		line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 

	return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
							   frames=frame_num, interval=10, blit=True)


plt.show()"""