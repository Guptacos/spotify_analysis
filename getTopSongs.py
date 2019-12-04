import billboard as bb
import pickle
import time


#for chart in bb.charts():
#	if chart.endswith("-songs"):
#		print(chart)
allCharts = []

#print(bb.ChartData('hot-100', date = "1900-01-01"))
chart = bb.ChartData('hot-100')
new = True
prevYear = chart.date.split('-')[0]
prevDate = None
while chart.date > '1999-12-31':
	if new:
		if prevDate != None and chart.date == prevDate:
			with open("hot100Charts"+prevYear, 'wb') as f:
				pickle.dump(allCharts, f)
				print("Checkpoint for Year: "+prevYear)
				exit()
		else:
			prevDate = chart.date

		currYear = chart.date.split('-')[0]
		if currYear != prevYear:
			with open("hot100Charts"+prevYear, 'wb') as f:
				pickle.dump(allCharts, f)
				print("Checkpoint for Year: "+prevYear)
			prevYear = currYear
			allCharts = []

		songs = []
		for song in chart.entries:
			songs.append({"name": song.title, "artist": song.artist, "date": chart.date})

		allCharts.append(songs)
	time.sleep(5)
	print(chart.date)
	try:
		chart = bb.ChartData('hot-100', chart.previousDate)
		new = True
	except Exception as e:
		print(type(e))
		new = False

with open("hot100Charts"+prevYear, 'wb') as f:
	pickle.dump(allCharts, f)
	print("Checkpoint for Year: "+prevYear)

print("Pickled")