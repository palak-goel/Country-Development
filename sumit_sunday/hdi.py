import pandas as pd
import itertools
import math
import numpy as np
import statistics
from collections import defaultdict
import statsmodels.formula.api as smf
import sys
import random
import csv

country_map = {}
#all_countries = {}
with open("sumit_sunday/Country.csv") as f:
	redr = csv.DictReader(f)
	for row in redr:
		country_map[row["ShortName"]] = row["CountryCode"]
		#all_countries[row["CountryCode"]] = len(all_countries)
print("HERE")
mat = {}
all_countries = set()
rdr = csv.DictReader(open("sumit_sunday/hdi.csv"))
yrs = list(rdr.fieldnames)
yrs.remove("Country")
yrs.remove("HDI Rank")
for row in rdr:
	for y in yrs:
		try:
			c = country_map[row["Country"].strip()]
			all_countries.add(c)
			mat[c, y] = row[y]
		except:
			pass
for y in yrs:
	wtr = csv.writer(open("sumit_sunday/hdi/hdi_" + str(y) + ".csv", "w"))
	wtr.writerow(["Country", "HDI"])
	for c in all_countries:
		if mat[c, y]:
			wtr.writerow([c, mat[c, y]])
		
