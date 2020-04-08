import csv, sys
import json
filename = 'sections.txt'

with open(filename, newline='',encoding="latin-1") as f:
    reader = csv.reader(f,delimiter="\t",quotechar='"')
    all_titles = []
    all_abtracts = []
    try:
        for row in reader:
        	if len(row) < 3: continue
        	if row[2] == "TITLE":
        		all_titles.append(row[3])
        		all_abtracts.append([])
        	elif len(all_titles) > 0: #if row[2] == "ABSTRACT":
        		all_abtracts[-1].append(row[3])

    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))


title_to_abstract = {}

for title,abstract in zip(all_titles,all_abtracts):
	title_to_abstract[title] = abstract

with open("titles_to_papers.json","w+") as json_file:
	json.dump(title_to_abstract,json_file)

