import numpy as np
import matplotlib.pyplot as plt
import math
import os

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
            print("warning, bad se: " + str(se_squared))
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

def write_subset_attrs(attr_codes, start=1964, end=2013):
    id_str = "_".join(attr_codes)
    csv_format = ",".join(attr_codes)
    for year in range(start,end+1):
        try:
            f = open(os.path.abspath("../data/mice_"+str(year) +".csv"), 'r')
            f_w = open(os.path.abspath("../data/subset/"+id_str+"_"+str(year) +".csv"), 'w+')
            idx_mapping = {}
            header = f.readline()
            for p, indicator in enumerate(header.split(",")):
                if p != 0:
                    idx_mapping[indicator.strip()] = p
            f_w.write("Country," + csv_format+'\n')
            for l in f: 
                l_tokenized = l.split(",")
                f_w.write(l_tokenized[0])
                for attr_code in attr_codes:
                    f_w.write("," + l_tokenized[idx_mapping[attr_code]])
                f_w.write("\n")
            f_w.close()
            f.close()
        except Exception as e:
            print(str(year)+" is bad because of " + e)
            continue

#this is pretty jank, i don't care
#aka don't use this for anything
def write_hdis(start=1964, end=2013):
    for year in range(start, end+1):
        try:
            f = open(os.path.abspath("../data/subset/SP.DYN.LE00.IN_NY.GDP.PCAP.CD_SE.PRM.ENRR_SE.ADT.LITR.ZS_"+str(year)+".csv"), 'r')
            f_w = open(os.path.abspath("../data/subset/hdi_"+str(year)+".csv"), 'w+')
            f_w.write("Country,LEI,GDP,EI,HDI\n")
            f.readline()
            for l in f: 
                l_tokenized = l.split(",")
                lei_raw, gdp_raw, gei_raw, ali_raw = float(l_tokenized[1]), float(l_tokenized[2]), float(l_tokenized[3]), float(l_tokenized[4])
                lei = (0 if lei_raw <= 0 else (lei_raw - 25) / 65)
                gdp = (0 if gdp_raw <= 0 else (math.log(gdp_raw - math.log(100)) / (math.log(40000)-math.log(100))))
                gei = (0 if gei_raw <= 0 else gei_raw/ 100)
                ali = (0 if ali_raw <= 0 else ali_raw/ 100)
                ei = 2*ali/3 + gei/3
                hdi = lei/3+gdp/3+ei/3
                f_w.write(l_tokenized[0]+","+str(lei)+","+str(gdp)+","+str(ei)+","+str(hdi)+"\n")
            f.close()
            f_w.close()
        except:
            print(str(year) + " is bad")
            continue

#makes files in interval [start, end]
#does the intermediate conversion to the db format.
#make sure the file names match the formats.
def make_pca_files(start, end):
    # for i in range(start, end+1):
    #     convert_mat_to_db(os.path.abspath("../data/compact/compact_" + str(i) + ".csv"), os.path.abspath("../data/mice_db_" + str(i) + ".csv"))
    for i in range(start, end+1):
        data, lcs = get_first_principal_components(os.path.abspath("../data/compact/compact_" + str(i) + ".csv"))
        reduced_mat, double_reduced_mat, ctys_in_order = generate_lc_mat(data, lcs)
        write_csv_from_mat(double_reduced_mat, ctys_in_order, os.path.abspath("../data/dr_mat_" + str(i) + "_mice.csv"))
        write_csv_from_mat(reduced_mat, ctys_in_order, os.path.abspath("../data/r_mat_" + str(i) + "_mice.csv"))

# make_pca_files(1995, 2012)

# np_mat, keys_used, u_reduce, imap = reduce_from_csv("recent_compact_2013.csv")
# write_csv_from_mat(np_mat, keys_used, "pca_2013.csv")
# write_pca_mat(u_reduce, imap, "pca_mat.csv")
# convert_mat_to_db("results.csv", "results_db_format.csv")
# imat, imap, keys_used = generate_data_matrix_for_imputation(create_map("recent_compact_2013.csv"))
# write_csv_from_mat_with_nulls(imat, imap, keys_used, "2013_mice.csv")

# write_subset_attrs(['SP.DYN.LE00.IN', 'NY.GDP.PCAP.CD', 'SE.PRM.ENRR', 'SE.ADT.LITR.ZS'])
write_hdis(1964,2013)


