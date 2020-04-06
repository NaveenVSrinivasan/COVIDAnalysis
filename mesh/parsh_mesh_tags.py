import xml.etree.ElementTree as ET

tree = ET.parse('desc2020.xml')
root = tree.getroot()

all_text = []


for child in root:
	for info in child:
		# print(info.tag)
		if info.tag == "DescriptorName":
			all_text.extend(info[0].text.lower().replace(",","").split())

clean_text = []

for t in all_text:
	clean_text.append(t)
	clean_text.append(t.replace("(","").replace(")",""))
	clean_text.extend(t.split("-"))
	clean_text.extend(t.replace("(","").replace(")","").split("-"))


clean_text = [x for x in clean_text if len(t) > 1 and not t.isdigit()]


for t in sorted(list(set(clean_text))):
	print(t)
	