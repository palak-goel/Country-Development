import pandas as pd

def read_csv_file():
	df = pd.read_csv("gni_test_file.csv")
	df['Year'] = df['Year'].astype(str)
	subset = df.loc[df['Year'] == "1960"]
	print(subset["Country"])
	print(subset["GNI"])

read_csv_file()