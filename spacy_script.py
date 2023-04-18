from src.models.spacy import SpacyModel
from src.utils import split_train_val
from src.data.spacy import SpacyDataLoader
from src.data.utils import read_dataset
from src.dataprocess import process
import json
import yake
from highlight import TextHighlighter
import pandas as pd
import os
from tqdm import tqdm
import glob


def eraseOverlapIntervals(intervals):
	intervals.sort()

	validIntervals = []
	validIntervals.append(intervals[0])
	n = len(intervals)
	lastEnd = intervals[0][1]
	for i in range(1, n):
		s, e = intervals[i]
		if s > lastEnd:
			lastEnd = e
			validIntervals.append(intervals[i])
		else:
			lastEnd = min(lastEnd, e)  # Choose the interval that has the shorter end point.

	return validIntervals


#data_proc = process("portuguese")	# remover o goiiiing print
#print("done preprocessing")

# if the script was never successfully run before
loaders = []
loadersval = []
if not os.path.isfile(r'/home/user/Downloads/yake-test/startendloaderstrain.txt'):
	#datafiles = ["/home/user/Downloads/yake-test/data/processed/portuguese/00000001.json"]
	path = "/home/user/Downloads/yake-test/data/processed/portuguese/"
	datafiles = os.listdir(path)

	list_of_files = sorted( filter( os.path.isfile, glob.glob(path + '*') ) )
	
	'''
	# ordered file read
	for data in list_of_files:
		#print(data)
		# Opening JSON file
		f = open(data)
		  
		# returns JSON object as 
		# a dictionary
		data = json.load(f)
		#print(data["text"])
		loaders.append(data["text"])
	'''
	
	# randomized file read
	ss = split_train_val(list_of_files)
	# ss[0] = train, ss[1] = validation
	
	for data in ss[0]:
		#print(data)
		# Opening JSON file
		f = open(data)
		  
		# returns JSON object as 
		# a dictionary
		data = json.load(f)
		#print(data["text"])
		loaders.append(data["text"])
		
	for data in ss[1]:
		#print(data)
		# Opening JSON file
		f = open(data)
		  
		# returns JSON object as 
		# a dictionary
		data = json.load(f)
		#print(data["text"])
		loadersval.append(data["text"])
		

	def weak_labelling(loaders, mode):
		actual_loaders = []
		startendloaders = []
		to_save = []

		# does weak labelling
		count = 0
		for text in tqdm(loaders):
			kw_extractor = yake.KeywordExtractor(lan="pt") # define the text language as portuguese
			keywords = kw_extractor.extract_keywords(text)

			#if count > 4:
			#	break
			#print(count+1)
			#ordered read: 1880? 11276? 12234? 14780? 16248? 20218? 21015? 25725? 35117? 38474?
			try:
				th = TextHighlighter(max_ngram_size = 3, highlight_pre = "", highlight_post= "")
				newtext, startends = th.highlight(text, keywords)
				new_list = []
				startends = set(startends) # remove duplicates
				startends = list(startends)
				startends = eraseOverlapIntervals(startends) # removes overlaps, doesnt work for all of them
				for tup in range(len(startends)):
					# add the label for fitting
					temp_list = list(startends[tup])
					temp_list.append("NOUN") # what label to use here?
					new_list.append(temp_list)
				startendloaders.append(new_list)
				actual_loaders.append(text)
				to_save.append([text, {"entities": new_list}])
				count = count + 1
				pass
			except:
				pass
			
		with open(r'/home/user/Downloads/yake-test/startendloaders'+mode+'.txt', 'w') as fp:
			for item in startendloaders:
			# write each item on a new line
				fp.write("%s\n" % item)
			print('Done start end values')
			
		loads = {}
		with open(r'/home/user/Downloads/yake-test/loaders'+mode+'.json', "w") as fp:
			idx = 0
			for item in actual_loaders:
			# write each item on a new line
				loads[idx] = item
				idx = idx + 1
			json.dump(loads, fp) 
			print('Done text values')
			
		print(to_save)
		with open(r'/home/user/Downloads/yake-test/loadersdict'+mode+'.json', "w") as fp:
	    		json.dump(to_save,fp) 
	    		print("Done json dict")
	    		
	weak_labelling(loaders,"train")
	weak_labelling(loadersval,"validation")
    		
    		
# if the script has been run before
actual_loaders = []
startendloaders = []
actual_loadersval = []
startendloadersval = []

with open(r'/home/user/Downloads/yake-test/startendloaderstrain.txt', 'r') as fp:
	for line in fp:
		# remove linebreak from a current name
		# linebreak is the last character of each line
		x = line[:-1]

		# add current item to the list
		startendloaders.append(x)
#print(startendloaders)

with open(r'/home/user/Downloads/yake-test/startendloadersvalidation.txt', 'r') as fp:
	for line in fp:
		# remove linebreak from a current name
		# linebreak is the last character of each line
		x = line[:-1]

		# add current item to the list
		startendloadersval.append(x)
#print(startendloaders)

f = open(r'/home/user/Downloads/yake-test/loaderstrain.json')
# returns JSON object as 
# a dictionary
loaders = json.load(f)
#print(loaders)
for _, x in loaders.items():
	actual_loaders.append(x)
#print(actual_loaders)

f = open(r'/home/user/Downloads/yake-test/loadersvalidation.json')
# returns JSON object as 
# a dictionary
loaders = json.load(f)
#print(loaders)
for _, x in loaders.items():
	actual_loadersval.append(x)
#print(actual_loaders)

f = open(r'/home/user/Downloads/yake-test/loadersdicttrain.json')
# returns JSON object as 
# a dictionary
loadersdict = json.load(f)
#print(loadersdict)

f = open(r'/home/user/Downloads/yake-test/loadersdictvalidation.json')
# returns JSON object as 
# a dictionary
loadersdictval = json.load(f)
#print(loadersdict)

model = SpacyModel("pt")

model.load("models/base/spacy/portuguese")

print("done loading model")

#val = model.predict(actual_loaders)
#print(val)

train_data = SpacyDataLoader(r'/home/user/Downloads/yake-test/loadersdicttrain.json')
eval_data = SpacyDataLoader(r'/home/user/Downloads/yake-test/loadersdictvalidation.json')
#print(data_fr.texts)
#print(data_fr.annotations)

#data_fr.format

#for batch in data_fr:
#	print(batch)
	
val = model.fit(train_data, eval_data, 10, 0.1, "/home/user/Downloads/yake-test/models/test")
#val = model.fit(data_fr, data_fr, 5, 0.1, "models/yake/portuguese/test")
print(val)

#val2 = model.evaluate(actual_loaders, startendloaders)
#print(val2)

with open(r'/home/user/Downloads/yake-test/fit.json', "w") as fp:
	json.dump(val,fp) 
	print("Done json dict")
	

# need to clean this up

