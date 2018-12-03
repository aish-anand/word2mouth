import spacy
import csv


output_dir = '/Users/mithramuthukrishnan/Documents/CS585'

print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

#read in csv with reviews
business = []
review = []
entities = []

with open('all_reviews.csv', newline = '') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	count = 0
	for row in reader:
		if count > 0:
			ents = []
			review.append(row[0])
			business.append(row[2])
			test_text = row[5]
			# print(row[5])
			doc = nlp2(test_text)
			for ent in doc.ents:
				ents.append((ent.text,ent.start_char,ent.end_char))
				# print(ent.label_,ent.text)
			entities.append(ents)
		count += 1
			
		



# write DATA to a csv file
with open('predictions.csv','w') as csvfile:
    fieldnames = ['review_id','business_id','entities']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(business)):
    	writer.writerow({'review_id':review[i], 'business_id':business[i], 'entities':entities[i]})

# doc2 = nlp2(test_text)
# for ent in doc2.ents:
#     print(ent.label_, ent.text)