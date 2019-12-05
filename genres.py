import billboard as bb

count = 0
for chart in bb.charts():
	# get only song charts. don't include greatest or streaming 
	if chart.endswith("-songs") and "greatest" not in chart and "streaming" not in chart:
		print(chart)
		count+=1

print(count)


# Picked these genres:

"""pop-songs,
country-songs,
rock-songs,
alternative-songs,
r-b-hip-hop-songs,
dance-electronic-songs,
christian-songs,
jazz-songs"""