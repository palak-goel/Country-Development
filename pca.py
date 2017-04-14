import numpy as np

COMPLETENESS = 0.75
VARIANCE_THRESHOLD = 0.95

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
    for key in data:
        metadata[key] = len(data[key].items())
    return metadata

def indicator_count_per_indicator(map_of_maps):
    metadata = {}
    for key in data:
        curr_map = data[key]
        for key in curr_map:
            metadata[key] = metadata.get(key, 0) + 1
    return metadata

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
    for key in map_of_maps:
        if icpc[key] > COMPLETENESS * num_indicators:
            curr_map = map_of_maps[key]
            row = [0 for i in range(num_indicators)]
            for indicator in curr_map:
                row[imap[indicator]] = curr_map.get(indicator, 0)
            info_matrix.append(row)
    return info_matrix

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

data = create_map("recent_compact_2013.csv")
icpc = indicator_count_per_country(data)
icpi = indicator_count_per_indicator(data)
imap = create_indicator_mapping(icpi)
mat = generate_data_matrix(data)
m = len(mat)
np_mat = np.matrix(mat)
sigma = np_mat.T * np_mat / m
u, s, v = np.linalg.svd(sigma)
u_reduce = pca(u, s, VARIANCE_THRESHOLD)
print(u_reduce.shape)

