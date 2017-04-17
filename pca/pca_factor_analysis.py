import pandas as pd

def read_csv():
	df = pd.read_csv('pca_mat.csv')
	df.convert_objects(convert_numeric=True)
	return df

def absolute_val():
	df = read_csv()
	for i in range(86):
		col = 'attr'+str(i)
		df[col] = df[col].abs()
	return df

def top_factors(nLargest, attribute):
	df = absolute_val()
	largest = df.nlargest(nLargest, attribute)
	original = read_csv()
	large_factors = largest['EMPTY']
	for index, factor in large_factors.iteritems():
		value = original.loc[original['EMPTY'] == factor][attribute]
		print(factor)
		print(value)


#input the nLargest factors, and the attribute
top_factors(5, 'attr1')