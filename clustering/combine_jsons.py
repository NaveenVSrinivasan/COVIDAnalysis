import json
import glob


file_suffix = "scibert_embeds_All.json"

all_data = {}

for file in sorted(list(glob.glob('*'+file_suffix))):
	with open(file,"r") as json_file:
		partial_json = json.load(json_file)
		for k,v in partial_json.items():
			all_data[k] = v

with open(file_suffix+"all","w+") as json_file:
	json.dump(all_data,json_file)