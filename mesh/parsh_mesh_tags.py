import xml.etree.ElementTree as ET

tree = ET.parse('desc2020.xml')
root = tree.getroot()

all_text = []


for child in root:
	for info in child:
		# print(info.tag)
		if info.tag == "DescriptorName":
			all_text.extend(info[0].text.lower().replace(",","").split())

for t in sorted(list(set(all_text))):
	print(t)
	