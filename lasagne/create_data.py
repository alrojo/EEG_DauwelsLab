import sys
import glob

import data

if len(sys.argv) != 2:
	sys.exit("Usage: python create_data <train/test>")
data_name = sys.argv[1]
assert data_name in ["train", "test"]

paths_csv = "./data/csv/" + data_name + "/*"
print("you csv path is: %s" % paths_csv)
print("Converting data ...")
data.convert_data(paths_csv)
