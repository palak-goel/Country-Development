import pandas as pd
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import OrderedDict
#import plotly.plotly as py
#import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import *
init_notebook_mode()
import numpy as np

countries = defaultdict(dict)
f = open('../gni_test_file.csv', 'r', encoding="utf8")
reader = csv.DictReader(f)

for dta in reader:
	ctry, yr, gni = dta["Country"], int(dta["Year"]), float(dta["GNI"])
	countries[ctry][yr] = gni
f.close()

keys = []
f = open('pca_2013.csv')
reader = csv.DictReader(f)
for dta in reader:
	keys.append(dta["Country"])
f.close()
'''
for ctry in keys:
	dta = countries[ctry]
	plt.plot(list(dta.keys()), list(dta.values()), label=ctry)
#plt.plot(gni)
#plt.xticks(range(len(countries)), countries)

plt.legend()
plt.savefig("GNI")
'''
to_plot=[]
x = []
y = []
slopes = {}
for ctry in keys:
	dta = countries[ctry]
	_x = list(dta.keys())
	_y = list(dta.values())
	tr = Scatter(
		x = _x,
		y = _y,
		name=ctry)
	#to_plot.append(tr)
	m, b = np.linalg.lstsq(np.vstack([_x, np.ones(len(_x))]).T, _y)[0]
	x.append(m)
	y.append(b)
	#to_plot.append(tr)
	slopes[ctry] = m,b

ord = OrderedDict(sorted(slopes.items(), key=lambda x: abs(x[1][0])))
print(ord)

obj = {"data": [Scatter(x=x, y=y)],"layout": Layout(title="hello world")}

#py.iplotx
plot(obj)