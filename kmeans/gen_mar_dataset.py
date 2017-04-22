import numpy as np

RANDOMNESS_THRESHOLD = 0.05

def gen_mar(filename, num_row, num_col):
    f = open(filename, 'w')
    rand _vals = np.random.rand(num_row, num_col)
    for i in range(num_row):
        for j in range(num_col):
            if rand_vals[i][j] > RANDOMNESS_THRESHOLD:
                if j == 0:
                    f.write(str(np.random.random_sample()))
                else:
                    f.write(","+str(np.random.random_sample()))
            elif j != 0:
                f.write(",")
        f.write("\n")

gen_mar("small_set.csv", 250, 1000)
