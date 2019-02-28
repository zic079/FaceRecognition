import glob
from shutil import copyfile

dirlist = glob.glob("*")

for dir in dirlist:
	filelist = glob.glob(dir + "/*")
	filecount = len(filelist)

	counter = 0

	for file in filelist:
		# 210 for training
		if(counter < 210):
			copyfile(file, "../train/" + file)
		
		# 30 for validation
		elif(counter < 240):
			copyfile(file, "../validation/" + file)
		
		# 60 for testing
		elif(counter < 300):
			copyfile(file, "../test/" + file)

		else:
			break

		counter += 1
