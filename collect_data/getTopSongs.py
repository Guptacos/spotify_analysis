import billboard as bb
import pickle
import time

allCharts = []

# get latest chart
chart = bb.ChartData('hot-100')
new = True
prevYear = chart.date.split('-')[0]

# songs thru 2000
while chart.date > '1999-12-31':
	# add week to list if new chart
	if new:
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
	try:
		# try to get previous week's chart, if this is last chart, break
		if not chart.previousDate:
			break
		chart = bb.ChartData('hot-100', chart.previousDate)
		new = True
	except Exception as e:
		# if fail from connection loss or too many requests, ignore and try again
		print(type(e))
		new = False

# save out last chart
with open("hot100Charts"+prevYear, 'wb') as f:
	pickle.dump(allCharts, f)
	print("Checkpoint for Year: "+prevYear)

print("Pickled")