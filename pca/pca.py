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
        se_squared = float(isspi[key] / float(icpi[key]) - (ispi[key] / float(icpi[key])) ** 2)
        if se_squared > 0:
            metadata[key] = math.sqrt(se_squared)
        #bandaid!!!!
        else:
            metadata[key] = 0
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

def generate_data_matrix_for_imputation(map_of_maps):
    icpc = indicator_count_per_country(map_of_maps)
    icpi = indicator_count_per_indicator(map_of_maps)
    imap = create_indicator_mapping(icpi)
    num_indicators = len(imap.keys())
    info_matrix = []
    keys_used = []
    for key in map_of_maps:
        keys_used.append(key)
        curr_map = map_of_maps[key]
        row = ["" for i in range(num_indicators)]
        for indicator in curr_map:
            row[imap[indicator]] = str(curr_map.get(indicator, ""))
        info_matrix.append(row)
    return info_matrix, imap, keys_used

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

def get_first_principal_components(filename):
    data = standardize_mapping(create_map(filename))
    icpi = indicator_count_per_indicator(data)
    imap = create_indicator_mapping(icpi)
    subsection_map = create_subsection_mapping(imap)
    means = indicator_mean_per_indicator(data)
    linear_combinations = {}
    for subsection in subsection_map:
        relevant_indicators = set(subsection_map[subsection])
        info_matrix, keys_used = [], []
        sub_imap = {}
        default_row = []
        count = 0
        for indicator in relevant_indicators:
            default_row.append(means[indicator])
            sub_imap[indicator] = count
            count += 1 
        for cty in data:
            cty_map = data[cty]
            row = default_row[:]
            for key in cty_map:
                if key in relevant_indicators:
                    row[sub_imap[key]] = cty_map[key]
            info_matrix.append(row)
        np_submat = np.matrix(info_matrix)
        m = len(np_submat)
        sigma = np_submat.T * np_submat / m
        u, s, v = np.linalg.svd(sigma)
        u_reduce = u[:,:1].T.tolist()
        u_reduce_list = [item for sublist in u_reduce for item in sublist]
        linear_comb = {}
        sub_imap_items = list(sub_imap.items())
        for p, ele in enumerate(u_reduce_list):
            linear_comb[sub_imap_items[p][0]] = ele
        linear_combinations[subsection] = linear_comb
    return data, linear_combinations

def generate_lc_mat(data, lcs):
    info_matrix = []
    attr_map = {}
    ctys_in_order = []
    means = indicator_mean_per_indicator(data)
    count = 0
    for lc in lcs: 
        attr_map[lc] = count
        count += 1
    for cty in data:
        ctys_in_order.append(cty)
        cty_record = data[cty]
        row = []
        for lc in lcs: 
            curr_lc = lcs[lc]
            attr = 0
            for key in curr_lc: 
                attr += cty_record.get(key, means[key])
            row.append(attr)
        info_matrix.append(row)
    np_mat = np.matrix(info_matrix)
    m = len(np_mat)
    sigma = np_mat.T * np_mat / m
    u, s, v = np.linalg.svd(sigma)

    #CHANGE LATER
    u_reduce = pca(u, s, VARIANCE_THRESHOLD)
    reduced_mat = np_mat * u_reduce
    return np_mat, reduced_mat, ctys_in_order


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

def write_csv_from_mat_with_nulls(imat, imap, keys_used, filename):
    f = open(filename, 'w')
    indicators_as_items = sorted(list(imap.items()), key=lambda x: x[1])
    row_length = len(indicators_as_items)
    default_row = ["" for i in range(row_length)]
    f.write("Country")
    for tup in indicators_as_items:
        f.write(","+tup[0])
    f.write("\n")
    for p, row in enumerate(imat):
        f.write(keys_used[p]+",")
        f.write(",".join(row)+"\n")
    f.close()

def create_subsection_mapping(imap):
    indicator_groups = {}
    for indicator in list(imap.keys()):
        group = indicator.split(".")[0]
        curr_grouping = indicator_groups.get(group, [])
        curr_grouping.append(indicator)
        indicator_groups[group] = curr_grouping
    return indicator_groups

def convert_mat_to_db(filename_r, filename_w):
    f_r = open(filename_r, 'r')
    f_w = open(filename_w, 'w')
    heading = f_r.readline().split(",")
    for line in f_r:
        csv_row_as_list = line.split(",")
        cty = csv_row_as_list[0]
        last_val = len(csv_row_as_list) - 1
        for p, ele in enumerate(csv_row_as_list):
            if p != 0 and p != last_val:
                f_w.write(cty+","+heading[p]+", "+ele+"\n")
    f_r.close()
    f_w.close()
# np_mat, keys_used, u_reduce, imap = reduce_from_csv("recent_compact_2013.csv")
# write_csv_from_mat(np_mat, keys_used, "pca_2013.csv")
# write_pca_mat(u_reduce, imap, "pca_mat.csv")
# convert_mat_to_db("results.csv", "results_db_format.csv")
'''for i in range(1960, 2014):
    name = str(i) + ".csv"
    data, lcs = get_first_principal_components("data/mice/mice_" + name)
    reduced_mat, double_reduced_mat, ctys_in_order = generate_lc_mat(data, lcs)
    write_csv_from_mat(double_reduced_mat, ctys_in_order, "data/full/full_dr_" + name)
    write_csv_from_mat(reduced_mat, ctys_in_order, "data/full/r_" + name)
#data, lcs = get_first_principal_components("../data/results_db_format.csv")'''
#reduced_mat, double_reduced_mat, ctys_in_order = generate_lc_mat(data, lcs)
#write_csv_from_mat(double_reduced_mat, ctys_in_order, "double_reduced_mat_2013_mice.csv")
#write_csv_from_mat(reduced_mat, ctys_in_order, "double_reduced_mat_2013_mice.csv")
for i in range(1960, 2014):
    imat, imap, keys_used = generate_data_matrix_for_imputation(create_map("data/compact_" + str(i) + ".csv"))
    write_csv_from_mat_with_nulls(imat, imap, keys_used, "data/full/full_" + str(i) + ".csv")


