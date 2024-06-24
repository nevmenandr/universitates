import os
import re

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

from pprint import pprint

import spacy

import pickle
import re
import pyLDAvis
import pyLDAvis.gensim_models

import matplotlib.pyplot as plt

###########################
#
#  git clone 
#  https://github.com/nevmenandr/universitates.git
#  https://github.com/nevmenandr/DigitalHumanitiesMinorFeatures.git
#
###########################

from pymystem3 import Mystem

m = Mystem()

stopdict = []
stpl = 'DigitalHumanitiesMinorFeatures/stop_ru.txt'
with open(stpl) as s:
	for w in s:
		stopdict.append(w.strip())
stopdict.extend(['какой-то', 'все-таки', 'поэтому', 'похоже', 'как-то',
                 'безусловно', 'ничто', 'кто-то', 'поскольку', 'нету', 
                 'собственно', 'что-то', 'вообще', 'по-моему', 'что-нибудь',
                 'где-то', 'либо', 'какой-нибудь'])


stopdict = set(stopdict)


pdir = 'universitates'
texts = []
for fl in os.listdir(pdir):
	if not fl.startswith('ep-'):
		continue
	print(fl)
	with open(os.path.join(pdir, fl)) as f:
		t = f.read()
	t = t.split('Транскрипт')[1]
	for paragraph in t.split('\n'):
		if not re.search('[а-я]+', paragraph):
			continue
		if re.search('^\\[.+?\\] — [а-яА-Я]+ [а-яА-Я]+\s?$', paragraph):
			continue
		paragraph = re.sub('\[[а-яА-Я]+ [а-яА-Я]+, 0:[0-9]{2}:[0-9]{2}\]', '', paragraph)
		lemmas = m.lemmatize(paragraph)
		p = []
		for l in lemmas:
			if not re.search('[а-я]+', l):
				continue
			if l in stopdict:
				continue
			if len(l) == 1:
				continue
			if l == 'курсы':
				l = 'курс'
			if l == 'орех':
				l = 'орехов'
			p.append(l)
		texts.append(p)
id2word = Dictionary(texts)    
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=10,
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'universitates/lda.html')

with open('universitates/lda.html') as f:
	html = f.read()

head1 = '''<!DOCTYPE html>
<html lang="en-US">
  <head>
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=640">
    <link rel="icon" type="image/x-icon" href="./favicon.ico">
<link '''

head2 = '''.css">
<title>Лига Айвы | Подкаст об университете</title>
  </head>
  <body>
	  <h1>Тематика подкаста «Лига Айвы»</h1>
	  
	  <p><a href="./">На главную</a></p>
	  '''
	  
foot = '''
  </body>
</html>'''

html = html.replace('<link ', head1)
html = html.replace('.css">', head2)

html += foot

with open('universitates/lda.html', 'w') as f:
	f.write(html)
