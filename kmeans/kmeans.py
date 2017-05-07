import numpy as np
import numpy.random
import csv
import matplotlib.pyplot as plt
from plotly.offline import plot 
import pandas as pd
import random
import threading 
from multiprocessing import Process
from multiprocessing import Manager

FILE = '../csv/reduced_mat_2013_mice.csv'
ATTRIBUTES = 0

"""
distance between two points
"""
def distance(A, B):
	return np.sum( (A-B)**2 )
"""
mean of an array
"""
def mean(data):
	attributes = len(data)
	#print(data)
	meanCoord = np.zeros(ATTRIBUTES, int)
	for point in data:
		meanCoord = meanCoord + np.array(point)
	return meanCoord/len(data)

"""
returns true if two arrays are equal
"""
def equals(A, B):
	for i in range(len(A)):
		if A[i] is not B[i]:
			return False
	return True

"""
calculates point scatter (error of cluster) MSE
"""
def scatter(data, means):
	keys = data.keys()
	distances = {}
	for k in keys:
		distances[k] = 0
	for k in keys:
		for point1 in data[k]:
			for point2 in data[k]:
				dist = distance(point1, point2)
				distances[k] += dist
	d = 0
	for k in distances:
		d += distances[k]
	return (0.5 * d)

def country_row(country, df):
	result = df.loc[(df["Country"] == country)]
	ans = result.values.T.tolist()
	final = []
	for i in range(1,len(ans)):
		a = ans[i]
		final.append(float(a[0]))
	return final

def avg_gni(countryList, yr):
	df = pd.read_csv('../gni_test_file.csv')
	#print(df)
	tot = 0
	totGni = 0
	for c in countryList:
		gni_data = df.loc[(df['Country'].astype(str) == c) & (df['Year'].astype(str) == yr)]
		gni = gni_data.values.T.tolist()[2]
		if (len(gni) > 0):
			print("HERE")
			tot += 1
			totGni += float(gni[0])
			print(tot)
			print(gni[0])
			print(totGni)
	print(countryList)
	print(yr)
	print(totGni)
	print(tot)
	return float(totGni/tot)


def cluster_by_GNI(clusters, yr):
	cluster_gni = []
	print(clusters)
	for c in clusters.keys():
		clust = clusters[c]
		tup = (clust, avg_gni(clust, yr))
		cluster_gni.append(tup)
	values = sorted(cluster_gni, key=lambda x: x[1])[::-1]
	cluster_gni_as_dict = {}
	ct = 0
	for t in values:
		cluster_gni_as_dict[ct] = t[0]
		ct+=1
	return cluster_gni_as_dict

def write_to_file(locations_dict, clusters_dict, yr, writer):
	for i in range(1,len(locations_dict.keys()) + 1):
		writer.writerow((yr, locations_dict[i], clusters_dict[i]))

#This is called on a clusters dictionary and creates a world map visualization
def country_graph(clusters, yr, writer):
	print("INPUT")
	print(clusters)
	clusters = cluster_by_GNI(clusters, yr)
	print("OUT")
	print(clusters)
	country_to_cluster = {}
	for k in clusters.keys():
		countries = clusters[k]
		for c in countries:
			country_to_cluster[c] = k

	locations_dict = {}
	clusters_dict = {}

	count = 0
	for k in country_to_cluster.keys():
		count = count + 1
		locations_dict[count] = k
		clusters_dict[count] = country_to_cluster[k]

	write_to_file(locations_dict, clusters_dict, yr, writer)

	scl = [[0.0, '#8CB369'],[0.2, '#B1DD83'],[0.4, '#F4E285'],\
            [0.6, '#F2CA7E'],[0.8, '#F78E69'],[1.0, 'rgb(214, 228, 225)']]

	df = pd.DataFrame({'Country' : locations_dict, 'Cluster' : clusters_dict})

	data = [dict( type = 'choropleth', locations = df['Country'],
			colorscale = scl,
        	autocolorscale = False,
            z = df['Cluster'],
            showscale = False)]

	layout = dict(title = 'Country Clusters '+yr, geo = dict(projection = dict(type = 'Mercator')))

	fig = dict(data=data, layout=layout)
	file_name = 'cluster'+yr+'.html'
	plot(fig, validate=True, filename=file_name)

#def initial_clusters(k, df):
#	countries = df['Country']


"""
K MEANS CLUSTERING
Input: number of clusters, dataframe
Output: cluster in the form of a dictionary
"""
def compute_cluster(clusters, df):
	acCountries = df['Country']
	acCountries = np.array(acCountries)
	X = df.ix[:, df.columns != 'Country']
	X = np.array(X)
	N = len(X)

	"""At this point data is loaded."""
	K = clusters

	aXmeans_byCluster = []

	"""Randomly select Countries"""
	countries = random_selection(df, clusters)

	for c in countries:
		row = country_row(c, df)
		aXmeans_byCluster.append(row)
	
	
	"""Set initial clusters (random)

	
	for i in range(K):
		cluster = []
		for j in range(1,ATTRIBUTES+1):
			maximum = max(df['x'+str(j)])
			minimum = min(df['x'+str(j)])
			r = random.uniform(minimum, maximum)
			cluster.append(r)
		aXmeans_byCluster.append(cluster)
	"""

	"""Set initial clusters (points)

	one = country_row("USA", df)
	two = country_row("NGA", df)
	three = country_row("IND", df)
	four = country_row("DEU", df)
	five = country_row("COL", df)

	aXmeans_byCluster.append(one)
	aXmeans_byCluster.append(two)
	aXmeans_byCluster.append(three)
	aXmeans_byCluster.append(four)
	aXmeans_byCluster.append(five)
	"""	



	converged = False
	#iterate until converged
	while not converged:

		cluster_assignment = {}
		cluster_data = {}

		for k in range(K):
			cluster_assignment[k] = []
			cluster_data[k] = []

		for n in range(N):
		    #Allocate each country to nearest cluster
		    country = acCountries[n]
		    data = X[n]

		    min_dist = 100000
		    min_index = 0
		    for k in range(K):
		    	dist = distance(data, aXmeans_byCluster[k])
		    	if dist < min_dist:
		    		min_dist = dist
		    		min_index = k

		    cluster_assignment[min_index].append(country)
		    cluster_data[min_index].append(data)

		# Compute within cluster point scatter
		count = 0
		for k in range(K):
		    # Compute cluster means
		    if (equals(aXmeans_byCluster[k],mean(cluster_data[k]))):
		    	count = count + 1
		    if (count == k):
		    	converged = True
		    aXmeans_byCluster[k] = mean(cluster_data[k])
	array = [] #[clusters, error]
	array.append(cluster_assignment)
	array.append(scatter(cluster_data, aXmeans_byCluster)) 
	return array

"""
runs compute cluster multiple times and returns the cluster with minimum error
K = the number of clusters
iterations = how many times to run the cluster
ans = dictionary to be returned
"""
def minimize(df, K, iterations, ans):
	prevMin = 10000000000
	for i in range(iterations): 
		array = compute_cluster(K, df)
		print(i)
		if (array[1] < prevMin):
			ans[0] = array[0]
			ans[1] = array[1]
			prevMin = array[1]
	print(prevMin)
	return prevMin

#This will create the graph we use to determine the number of clusters using elbow
def graph(df, iterations):
	xVal = []
	yVal = []
	for i in range(1,16):
		print(i)
		xVal.append(i)
		yVal.append(threading(df, i, iterations)[1])
	print(xVal)
	print(yVal)
	plt.plot(xVal, yVal)
	plt.ylabel('Cluster Point Scatter')
	plt.xlabel('Number of Clusters')
	plt.show()


#Concurrently calls the minimize() function to speed it up, returns cluster dictionary	
def threading(df, k, iterate):
	manager = Manager()
	ans1 = manager.list(range(2))
	ans2 = manager.list(range(2))
	print(iterate/2)
	p1 = Process(target = minimize, args = (df, k,int(iterate/2), ans1))
	p2 = Process(target = minimize, args = (df, k,int(iterate/2), ans2))

	p1.start()
	p2.start()
	p1.join()
	p2.join()

	answers = []
	answers.append(ans1)
	answers.append(ans2)

	prevMin = 10000000000
	clusters = {}
	for a in answers:
		if a[1] < prevMin:
			prevMin = a[1]
			clusters = a[0]
	print(prevMin)
	return (clusters, prevMin)

def random_selection(df, numClusters):
	results = []
	countries = df['Country']
	numCountries = len(countries)
	for i in range(numClusters):
		random_number = int(random.uniform(0, numCountries))
		country = countries.ix[[random_number]]
		results.append(country.values.T.tolist())
	ans = []
	for c in results:
		ans.append(c[0])
	return ans

def get_clusters(numClusters, csvFile, iterations, mode, year, writer):
	df = pd.read_csv(csvFile)
	global ATTRIBUTES
	ATTRIBUTES = len(df.columns) - 1
	if mode == 'graph':
		graph(df, iterations)
	if mode == 'cluster':
		cluster = threading(df, numClusters, iterations)[0]
		country_graph(cluster, year, writer)

def run():
	csvfile = open('test.csv', 'w')
	writer = csv.writer(csvfile)
	writer.writerow( ('Year', 'Country', 'Cluster') )

	for i in range(1988,2013):
		path = '../data/subset/hdi_' + str(i) + ".csv"
		get_clusters(3, path, 100, 'cluster', str(i), writer)

run()
#get_clusters(3, FILE, 100, 'cluster', '2013')
