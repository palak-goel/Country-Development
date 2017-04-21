import pandas as pd
import itertools
import math
import numpy as np
import statistics
from collections import defaultdict
import statsmodels.formula.api as smf
import sys
import random

#number of cols to do OLS on
SUBSET_SIZE = 200

# Returns a function that produces a random value from a normal distribution as 
# specified with the given data
def gen_distr_producer(d):
	dta = list(filter(lambda a: not math.isnan(a), list(d)))
	#print(dta)
	std = np.std(dta)
	m = np.mean(dta)
	if std == 0.0:
		return lambda: m
	else:
		return lambda: np.random.normal(m, std)

def within(v1, v2):
	return abs(v1 - v2) < 2

# Termination condition between the iteration and previous
def done(d, d_prev, columns, to_check):
	#TODO
	flag = True
	for c in columns:
		new_idx = list(to_check[c])
		for i in to_check[c]:
			if not within(d.ix[i, c], d_prev.ix[i, c]):
				flag = False
				new_idx.append(i)
			to_check[c] = new_idx
	return flag

# Initial normal distribution values for NaN (missing) values
def gen_nans(df, cols, to_check):
	ctr = 0
	for c in cols:
		ctr +=1
		if ctr % 50 == 0:
			print("{0} done with normal distr".format(ctr))
		n_dist = gen_distr_producer(df[c])
		for i in to_check[c]:
			df.ix[i, c] = n_dist()
		#gen_ols(df, c, to_check[c])

# Predict given column and substitute the values in "to check" with predicted
def gen_ols(df, cols, to_check):
	ctr = 0
	for c in cols:
		ctr +=1
		if ctr % 50 == 0:
			print("{0} done with ols".format(ctr))
		params = "+".join(random.sample(cols, SUBSET_SIZE))
		f = c + "~" + params + "-" + c
		pred = smf.ols(formula = f, data=df).fit().predict()
		for i in to_check[c]:
			df.ix[i, c] = pred[i]

#setting up
df = pd.read_csv("2013_mice.csv")
df.dropna(how="all", inplace=True)

#map of col names for printing properly later
col_map = {}
for c in df.columns:
	col_map[c.strip().replace(".", "")] = c
df.rename(columns=lambda x: x.strip().replace(".", ""), inplace=True)
columns = list(df.columns)
columns.remove("Country")
#parameters (columns to regress on) need to be randomized, fixed rn
#params = "+".join(columns[:SUBSET_SIZE]) 
rows, cols = df.shape
na_vals = defaultdict(list)

# getting all the missing values, putting them in a hash from column -> row index
for c, v in itertools.product(columns, range(rows)):
	if math.isnan(df[c][v]):
		na_vals[c].append(v)
#iteration

#still needs to be fleshed out more
for _ in range(1):
	df_prev = df.copy()
	gen_nans(df, columns, na_vals)
	gen_ols(df, columns, na_vals)
	print("---")
	print("iter done")
	print("---")
	#if done(df, df_prev, columns, na_vals):
	#	break
#while not_done(df, df_prev):
#	gen_ols(df, columns, na_vals)

df.rename(columns=lambda x: col_map[x], inplace=True)
df.to_csv("results.csv", index=False)






