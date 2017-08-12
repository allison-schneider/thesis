import os

for f in os.listdir("."):
	os.rename(f, f.replace("test_",""))