import pandas as pd
import itertools
import math
import numpy as np
import statistics
from collections import defaultdict, Counter
import statsmodels.formula.api as smf
import sys
import random
import csv

folders = ["r_clusters", "hdi_cluster", "dr_cluster", "gni_cluster"]

def analyze(fName):
	path = "kmeans/" + fName + "/test.csv"
	rdr = csv.DictReader(open(path, "r"))
	similar_to_country = defaultdict(Counter)
	country_clusters = defaultdict(list)
	countries = set()
	prev_yr = None
	clusters = {0: [], 1: [], 2: []}
	for row in rdr:
		countries.add(row["Country"])
		if prev_yr and not (row["Year"] == prev_yr):
			for c in countries:
				cluster = list(clusters[country_clusters[c][-1]])
				cluster.remove(c)
				ctr = similar_to_country[c]
				for other_c in cluster:
					ctr[other_c] += 1
			clusters = {0: [], 1: [], 2: []}
		country_clusters[row["Country"]].append(int(row["Cluster"]))
		clusters[int(row["Cluster"])].append(row["Country"])
		prev_yr = row["Year"]
		print(prev_yr)
	wtr = csv.writer(open("kmeans/" + fName + "/averages.csv", "w"))
	wtr2 = csv.writer(open("kmeans/" + fName + "/similarities.csv", "w"))
	wtr.writerow(["Country", "Average", "Stdev"])
	wtr2.writerow(["Country", "Nbr1", "Nbr2", "Nbr3"])
	for c in countries:
		wtr.writerow([c, np.mean(country_clusters[c]), np.std(country_clusters[c])])
		nbrs = list(map(lambda x: x[0], similar_to_country[c].most_common(3)))
		wtr2.writerow([c] + nbrs)
for f in folders:
	analyze(f)


