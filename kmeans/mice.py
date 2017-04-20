import pandas as pd
import itertools
import math
import numpy as np
import statistics
from collections import defaultdict
import statsmodels.formula.api as smf

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

# Termination condition between the iteration and previous
def not_done(d, d_prev):
	#TODO
	return True

# Initial normal distribution values for NaN (missing) values
def gen_nans(df, cols, to_check):
	for c in cols:
		print(c)
		n_dist = gen_distr_producer(df[c])
		for i in to_check[c]:
			df.ix[i, c] = n_dist()
		#gen_ols(df, c, to_check[c])

# Predict given column and substitute the values in "to check" with predicted
def gen_ols(df, col, to_check):
	f = col + "~" + params + "-" + col
	print(col)
	pred = smf.ols(formula = f, data=df).fit().predict()
	for i in to_check:
		df.ix[i, c] = pred[i]

#setting up
df = pd.read_csv("2013_mice.csv")
df.dropna(how="all", inplace=True)
df.rename(columns=lambda x: x.strip().replace(".", ""), inplace=True)
columns = list(df.columns)
columns.remove("Country")
#parameters (columns to regress on) need to be randomized, fixed rn
params = "+".join(columns[:SUBSET_SIZE]) 
rows, cols = df.shape
na_vals = defaultdict(list)

# getting all the missing values, putting them in a hash from column -> row index
for c, v in itertools.product(columns, range(rows)):
	if math.isnan(df[c][v]):
		na_vals[c].append(v)

#iteration

#still needs to be fleshed out more

df_prev = df.copy()
gen_nans(df, columns, na_vals)
#while not_done(df, df_prev):
#	gen_ols(df, columns, na_vals)






