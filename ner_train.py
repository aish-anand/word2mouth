from __future__ import unicode_literals, print_function
from collections import Counter
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.gold import offsets_from_biluo_tags
import json
import re
import glob
import json
import random
import csv
from spacy.gold import GoldParse
from spacy.scorer import Scorer
glob_data = []
glob_data
# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
SEED = 50
DATA = []
#f = open('dataFinal.txt','w')
count = 0
for file in glob.glob('*.json'):
    #print(file)
    name = file.split(".ann")[0]
    #print(name)
    text_file = name + ".txt"
    if(text_file != ""):
        t_file = open(text_file,"r",encoding = "utf8")
#         print("json filename: " + text_file)
#         print("raw text of review: " + t_file.read())


    
    review = t_file.read()
    entities = []
    with open(name+'.ann.json') as json_file:
        data = json.load(json_file)
        for r in data['entities']:
    #         print('Full',r['offsets'])
    #         print('Index',r['offsets'][0]['start'])
    #         print('Food',r['offsets'][0]['text'])
            foodstring = r['offsets'][0]['text']
            l = len(foodstring)
            i = int(r['offsets'][0]['start']) #used to keep track of position
            entity = (i, i+l, 'FOOD')
            #print(entity)
            entities.append(entity)
            count += 1
            #reviews[r['offsets'][0]['start']] = [r['offsets'][0]['text']]
    DATA.append((review,{'entities':entities}))


# # write DATA to a csv file
# with open('annotationsNER.csv','w') as csvfile:
#     fieldnames = ['review','entities']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for x in DATA:
#         writer.writerow({'review':x[0], 'entities':x[1]['entities']})





def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        #print("gold:",doc_gold_text)
        # print(input_,annot)
        # print(annot['entities'])
        
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        print("Gold: ",gold.labels, gold.cats, gold.ents)
        pred_value = ner_model(input_)
        print("Pred: ",pred_value.ents)
        scorer.score(pred_value.ents, gold)
    return scorer.scores
# output




LABEL = 'FOOD'

print(len(DATA))

random.seed(SEED)
random.shuffle(DATA)

TRAIN_DATA = DATA[:80]


print(len(TRAIN_DATA))
TEST_DATA = DATA[80:]
print(len(TEST_DATA))


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, new_model_name='animal', output_dir=None, n_iter=10):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 35., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                # print("Text:",texts)
                # print("Annotations:",annotations)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # test the trained model
    # doc = []
    # golds = []
    # for input_, annot in TEST_DATA:
    #     doc.append(nlp.make_doc(input_))
    #     golds.append(annot['entities'])

    # docs = []
    # golds = []
    # scorer = Scorer()
    # for row in TEST_DATA:
    #     docs.append(nlp.make_doc(row[0]))
    #     golds.append(row[1]['entities'])
    # print("golds[0]: ",golds[0])
    # print("golds[0]: ",golds[1])
    # for name, pipe in nlp.pipeline:
    #     docs = (pipe(doc) for doc in docs)
    # for doc, gold in zip(docs,golds):
    #     print(gold)
    #     print("length: ",len(gold))
    #     scorer.score(doc, gold)
 
    meanprecision = 0
    TP = 0
    FP = 0
    FN = 0
    gold_total = 0
    for inputs, annot in TEST_DATA:
        gold = Counter()
        doc = nlp(inputs)
        entities = annot['entities']
        for x in entities:
            w = inputs[x[0]:x[1]]
            gold_total += 1
            gold[w] += 1
        print("Gold:",gold)
        print("Document",doc)
        for ent in doc.ents:
            print("Predictions",ent.text,ent.start_char,ent.end_char)
            if ent.text in gold and gold[ent.text] > 0:
                TP += 1
                gold[ent.text] -= 1
            elif ent.text not in gold or gold[ent.text] == 0:
                FP += 1
    FN = gold_total - TP
    precision = TP / (TP + FP + 1e-100)
    recall = TP / (TP + FN + 1e-100)
    print("Precision: ",precision,"Recall: ",recall,'FN',FN,'TP',TP)




# end of code
    # meanprecision += precision
    # meanprecision = meanprecision/len(TEST_DATA)
    # print(meanprecision)
        # print("Gold:",gold,"\nPredicted:",predicted)
        # print(TP,FP, gold_total-TP)







#scoring metric
    # precision = 0
    # for inputs, annot in TEST_DATA:
    #     gold = Counter()
    #     predicted = 0
    #     relevant = 0
    #     doc = nlp(inputs)
    #     entities = annot['entities']
    #     total = 0
    #     #true values
    #     for x in entities:
    #         i = x[0]
    #         j = x[1]
    #         w = inputs[i:j]
    #         total += 1
    #         gold[w] += 1
    #     # print(gold)
    #     # print(entities)
    #     for ent in doc.ents:
    #         relevant += 1
    #         if ent.text in gold and gold[ent.text] > 0:
    #             print('ent:', ent.text)
    #             gold[ent.text] -= 1
    #             predicted += 1
    #     print(predicted/total)
    #     precision += predicted/total
    # print(precision/len(TEST_DATA))
            #print(ent.label_,ent.text)

        # print("Gold:",gold,"\nPredicted",predicted)





    # scorer = Scorer()
    # for input_,annot in TEST_DATA:
    #     doc_gold_text = nlp.make_doc(input_)
    #     gold = GoldParse(doc_gold_text, )



    # golddocs = zip(doc,golds)
    # print("Length of docgolds:",len(golds),len(doc))
    # r = nlp.evaluate(golddocs)
    # print(r)

        
    

        #print("gold:",doc_gold_text)
        # print(input_,annot)
        # print(annot['entities'])
        
        # gold = GoldParse(doc_gold_text, entities=annot['entities'])
        # print("Gold: ",gold)
        # pred_value = ner_model(input_)
        # print("Pred: ",pred_value)
        # scorer.score(pred_value, gold)

    # test_text = 'Do you like scrambled eggs, corned beef, or waffle and berry?'
    # doc = nlp(test_text)
    # print(doc)
    # print("Entities in '%s'" % test_text)
    # print(doc.ents)
    # for ent in doc.ents:
    #     print(ent.label_, ent.text)
    # results = nlp.evaluate(TEST_DATA)
    # print(results)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)