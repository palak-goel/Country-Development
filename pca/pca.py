import numpy as np
import matplotlib.pyplot as plt
import math

#parameters to tune
COMPLETENESS = 0.50
VARIANCE_THRESHOLD = 0.95

#functions to use: 
#reduce_from_csv(filename) will take in a csv and create a PCA reduced one. 
#it returns the reduced matrix and the keys used (i.e. which countries weren't thrown out).
#write_csv_from_mat(np_mat, keys_used, filename) takes in the output of reduce_from_csv as
#the first two args, and whatever you want to call the file as the third arg and writes
#the reduced data to that file

def create_map(filename):
    file_to_read = open(filename, 'r')
    map_of_maps = {}
    for p, line in enumerate(file_to_read):
        record = line.split(",")
        record_as_dict = map_of_maps.get(record[0], {})
        record_as_dict[record[1]] = float(record[2])
        map_of_maps[record[0]] = record_as_dict
    return map_of_maps

def indicator_count_per_country(map_of_maps):
    metadata = {}
    for key in map_of_maps:
        metadata[key] = len(map_of_maps[key].items())
    return metadata

def indicator_count_per_indicator(map_of_maps):
    metadata = {}
    for key in map_of_maps:
        curr_map = map_of_maps[key]
        for key in curr_map:
            metadata[key] = metadata.get(key, 0) + 1
    return metadata

def indicator_sum_per_indicator(map_of_maps):
    metadata = {}
    for key in map_of_maps:
        curr_map = map_of_maps[key]
        for key in curr_map:
            metadata[key] = metadata.get(key, 0) + curr_map[key]
    return metadata

def indicator_sum_squared_per_indicator(map_of_maps):
    metadata = {}
    for key in map_of_maps:
        curr_map = map_of_maps[key]
        for key in curr_map:
            metadata[key] = metadata.get(key, 0) + curr_map[key] ** 2
    return metadata

def indicator_mean_per_indicator(map_of_maps):
    metadata = {}
    icpi, ispi = indicator_count_per_indicator(map_of_maps), indicator_sum_per_indicator(map_of_maps)
    for key in icpi:
        metadata[key] = ispi[key] / float(icpi[key])
    return metadata

def indicator_stdev_per_indicator(map_of_maps):
    metadata = {}
    icpi, ispi, isspi = indicator_count_per_indicator(map_of_maps), indicator_sum_per_indicator(map_of_maps), indicator_sum_squared_per_indicator(map_of_maps)
    for key in icpi:
        metadata[key] = math.sqrt(isspi[key] / float(icpi[key]) - (ispi[key] / float(icpi[key])) ** 2)
    return metadata

def standardize_mapping(map_of_maps):
    map_of_maps_standard = {}
    impi, istdevpi = indicator_mean_per_indicator(map_of_maps), indicator_stdev_per_indicator(map_of_maps)
    for key in map_of_maps:
        curr_map = map_of_maps[key]
        curr_map_standard = {}
        for attribute in curr_map:
            if istdevpi[attribute] != 0:
                curr_map_standard[attribute] = (curr_map[attribute] - impi[attribute]) / istdevpi[attribute]
        map_of_maps_standard[key] = curr_map_standard
    return map_of_maps_standard

def create_indicator_mapping(icpi):
    imap = {}
    for p, key in enumerate(icpi.keys()):
        imap[key] = p
    return imap

def generate_data_matrix(map_of_maps):
    icpc = indicator_count_per_country(map_of_maps)
    icpi = indicator_count_per_indicator(map_of_maps)
    imap = create_indicator_mapping(icpi)
    num_indicators = len(imap.keys())
    info_matrix = []
    keys_used = []
    for key in map_of_maps:
        if icpc[key] > COMPLETENESS * num_indicators:
            keys_used.append(key)
            curr_map = map_of_maps[key]
            row = [0 for i in range(num_indicators)]
            for indicator in curr_map:
                row[imap[indicator]] = curr_map.get(indicator, 0)
            info_matrix.append(row)
    return info_matrix, keys_used

def find_k(s, threshold):
    #min k st. retained var > threshold
    tot = np.sum(s)
    for i in range(1, s.size):
        curr = np.sum(s[:i])
        if curr / tot > threshold:
            return i
    return s.size - 1

def pca(u, s, threshold):
    k = find_k(s, threshold)
    return u[:, :k]

def reduce_from_csv(filename):
    data = create_map(filename)
    icpc = indicator_count_per_country(data)
    icpi = indicator_count_per_indicator(data)
    data_standard = standardize_mapping(data)

    imap = create_indicator_mapping(icpi)
    mat, keys_used = generate_data_matrix(data_standard)
    m = len(mat)
    np_mat = np.matrix(mat)

    sigma = np_mat.T * np_mat / m
    u, s, v = np.linalg.svd(sigma)
    u_reduce = pca(u, s, VARIANCE_THRESHOLD)
    reduced_mat = np_mat * u_reduce
    return reduced_mat, keys_used, u_reduce, imap

def write_pca_mat(u_reduce, imap, filename):
    f = open(filename, 'w')
    imap_items = list(imap.items())
    imap_items.sort(key=lambda x: x[1])
    num_cols = u_reduce.shape[1]
    f.write("EMPTY")
    for i in range(num_cols):
        f.write(",attr" + str(i))
    f.write("\n")
    for p, row in enumerate(u_reduce):
        f.write(imap_items[p][0])
        for ele in np.nditer(row): 
            f.write(","+str(ele))
        f.write("\n")
    f.close()

def write_csv_from_mat(np_mat, keys_used, filename):
    f = open(filename, 'w')
    num_cols = np_mat.shape[1]
    f.write("Country")
    for i in range(num_cols):
        f.write(",x"+str(i+1))
    f.write("\n")
    for p, row in enumerate(np_mat):
        f.write(keys_used[p])
        for ele in np.nditer(row):
            f.write(","+str(ele))
        f.write("\n")
    f.close()    

np_mat, keys_used, u_reduce, imap = reduce_from_csv("recent_compact_2013.csv")
write_csv_from_mat(np_mat, keys_used, "pca_2013.csv")
write_pca_mat(u_reduce, imap, "pca_mat.csv")

