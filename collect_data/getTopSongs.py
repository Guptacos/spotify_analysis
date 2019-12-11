import billboard as bb
import pickle
import time
import requests


def getNextChart(previousDate):
	try:
		# try to get previous week's chart, if this is last chart, break
		chart = bb.ChartData('hot-100', previousDate)
		return chart
	except requests.exceptions.ConnectionError as e:
		print(type(e))
		return getNextChart(previousDate)
	except Exception as e:
		# if fail from connection loss or too many requests, ignore and try again
		print(type(e))
		time.sleep(60)
		return getNextChart(previousDate)


allCharts = []

# get latest chart
chart = bb.ChartData('hot-100')
prevYear = chart.date.split('-')[0]

# songs thru 2000
while chart.date > '1969-12-31':
	# add week to list if new chart
	currYear = chart.date.split('-')[0]

	# checkpoint at the end of each year
	if currYear != prevYear:
		with open("hot100Charts"+prevYear, 'wb') as f:
			pickle.dump(allCharts, f)
			print("Checkpoint for Year: "+prevYear)
		prevYear = currYear
		allCharts = []

	# add week's songs to the list
	songs = []
	for song in chart.entries:
		songs.append({"name": song.title, "artist": song.artist, "date": chart.date})

	allCharts.append(songs)

	# wait 5 seconds and try to get another chart to avoid overwhleming billboard servers
	time.sleep(5)
	print(chart.date)
	if not chart.previousDate:
		break
	
	chart = getNextChart(chart.previousDate)

# save out last chart
with open("hot100Charts"+prevYear, 'wb') as f:
	pickle.dump(allCharts, f)
	print("Checkpoint for Year: "+prevYear)

print("Pickled")