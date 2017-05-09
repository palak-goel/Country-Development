import numpy as np
import matplotlib.pyplot as plt
import math
import os

#requires mice/subset files.
def calc_err(base_filename, start=2008,end=2010, l2_norm=True):
    mses = {}
    for year in range(start,end+1):
        try:
            f_predict = open(base_filename+str(year)+".csv", 'r')
            f_actual = open("NY.GNP.PCAP.CD_"+str(year)+".csv", 'r')
            f_predict.readline()
            f_actual.readline()
            err, count = 0, 0
            for l1, l2 in zip(f_predict, f_actual):
                predict, actual = float(l1.split(",")[1].strip()), float(l2.split(",")[1].strip())
                if l2_norm:
                    err += (predict-actual) ** 2
                else:
                    err += abs(predict-actual)
                count += 1
            mses[year] = err / count
        except:
            print(str(year)+" is bad")
            continue
    return mses

print(calc_err("dr_vals_", l2_norm=False))
print(calc_err("r_vals_", l2_norm=False))
print(calc_err("hdi_whole_", l2_norm=False))
print(calc_err("hdi_components_", l2_norm=False))
print(calc_err("dr_vals_"))
print(calc_err("r_vals_"))
print(calc_err("hdi_whole_"))
print(calc_err("hdi_components_"))
