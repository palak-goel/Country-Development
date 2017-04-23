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


ATTRIBUTES = 6

"""
distance between two points
"""
def distance(A, B):
	return np.sum( (A-B)**2 )
"""
mean of an array
"""
def mean(data):
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

	"""Set initial clusters (points)	"""

	usa = country_row("USA", df)
	gbr = country_row("GBR", df)
	ago = country_row("AGO", df)
	kgz = country_row("KGZ", df)
	bra = country_row("BRA", df)


	aXmeans_byCluster.append(usa)
	aXmeans_byCluster.append(gbr)
	aXmeans_byCluster.append(ago)
	aXmeans_byCluster.append(kgz)
	aXmeans_byCluster.append(bra)


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
def minimize(K, iterations, ans):
	prevMin = 10000000000
	df = pd.read_csv('../csv/double_reduced_mat_2013_mice.csv')
	for i in range(iterations): 
		array = compute_cluster(K, df)
		print(i)
		if (array[1] < prevMin):
			ans[0] = array[0]
			ans[1] = array[1]
			prevMin = array[1]
	#country_graph(ans)
	print(prevMin)
	return prevMin

#This will create the graph we use to determine the number of clusters using elbow
def graph():
	xVal = []
	yVal = []
	for i in range(1,16):
		print(i)
		xVal.append(i)
		yVal.append(threading(i, 100))
	print(xVal)
	print(yVal)
	plt.plot(xVal, yVal)
	plt.ylabel('Cluster Point Scatter')
	plt.xlabel('Number of Clusters')
	plt.show()


#Concurrently calls the minimize() function to speed it up, returns cluster dictionary	
def threading(k, iterate):
	manager = Manager()
	ans1 = manager.list(range(2))
	ans2 = manager.list(range(2))
	print(iterate/2)
	p1 = Process(target = minimize, args = (k,int(iterate/2), ans1))
	p2 = Process(target = minimize, args = (k,int(iterate/2), ans2))

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
	return clusters
	#return(prevMin)

#graph()
#c = threading(2, 1000)
c = compute_cluster(5, pd.read_csv('../csv/double_reduced_mat_2013_mice.csv'))
country_graph(c[0])


