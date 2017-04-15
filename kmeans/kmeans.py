import numpy as np
import numpy.random
import csv
import matplotlib.pyplot as plt
from plotly.offline import plot 
import pandas as pd
import random

#distance between two points
def distance(A, B):
	return np.sum( (A-B)**2 )

#mean of an array
def mean(data):
	meanCoord = np.zeros(86, int)
	for point in data:
		meanCoord = meanCoord + np.array(point)
	return meanCoord/len(data)

#error
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

#returns true if two arrays are equal
def equals(A, B):
	for i in range(len(A)):
		if A[i] is not B[i]:
			return False
	return True

# K MEANS CLUSTERING
def compute_cluster(clusters):
	# READ IN DATA
	f = open("pca_2013.csv", "rt", encoding="utf8")
	csvReader  = csv.DictReader(f)
	acCountries = []
	X = []

	#acCountries contains all of the countries
	#X contains all of the attributes
	row = 0
	for cLine in csvReader:
		acCountries.append(cLine["Country"])
		attributes = []
		for attr in cLine:
			try: 
				attributes.append(float(cLine[attr]))
			except ValueError:
				continue
				
		X.append(attributes)

	acCountries = np.array(acCountries)
	X = np.array(X)
	N = len(X)


	#At this point data is loaded.
	K = clusters

	# set initial clusters 

	df = pd.read_csv('pca_2013.csv')
	aXmeans_byCluster = []
	for i in range(K):
		cluster = []
		for j in range(1,87):
			maximum = max(df['x'+str(j)])
			minimum = min(df['x'+str(j)])
			r = random.uniform(minimum, maximum)
			cluster.append(r)
		aXmeans_byCluster.append(cluster)
	
	###################

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

#runs compute cluster 100,000 times and returns the cluster with minimum error
#K = the number of clusters
def minimize(K):
	ans = {}
	prevMin = 10000000
	for i in range(100000): #    <----  MODIFY THIS NUMBER 
		array = compute_cluster(K)
		if (array[1] < prevMin):
			prevMin = array[1]
			ans = array[0]
	print(ans)
	country_graph(ans)
	return prevMin

#This will create the graph we use to determine the number of clusters using elbow
def graph():
	xVal = []
	yVal = []
	for i in range(1,21):
		print(i)
		xVal.append(i)
		yVal.append(minimize(i))
	print(xVal)
	print(yVal)
	plt.plot(xVal, yVal)
	plt.ylabel('Cluster Point Scatter')
	plt.xlabel('Number of Clusters')
	plt.show()

#This is called on a clusters dictionary and creates a world map visualization
def country_graph(clusters):
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

	df = pd.DataFrame({'Country' : locations_dict, 'Cluster' : clusters_dict})

	data = [dict( type = 'choropleth', locations = df['Country'],
            z = df['Cluster'],
            colorbar = dict(title = 'Cluster'))]

	layout = dict(title = 'Country Clusters', geo = dict(projection = dict(type = 'Mercator')))

	fig = dict(data=data, layout=layout)
	plot(fig, validate=True, filename='count-cluster')
	

print(minimize(6))
#graph()

