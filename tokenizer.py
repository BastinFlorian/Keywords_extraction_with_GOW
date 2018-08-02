import spacy
import re
import os
import sys
import textacy
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#sys.argv[1]: root directory of extracted files
#sys.argv[2]: root directory of tokenized files

french_stemmer = SnowballStemmer('french')
nlp = spacy.load('/home/data/spacy/fr_core_news_sm-2.0.0/fr_core_news_sm/fr_core_news_sm-2.0.0')
nlp.max_length = 4*nlp.max_length

def line_cleaner(sentence):
    sentence = sentence.replace("  "," ")
    return re.sub(r"[\n]", r"", sentence)

# GENERATE THE LIST OF ALL EXTRACTED FILES PATHS
def path_generator(initial_root):
    all_paths = []
    left_paths = []
    #TXT or CSV
    for root, dirs, files in os.walk(initial_root):
        for name in files:
            all_paths.append(os.path.join(root, name))
    begin = False
    exp = re.compile('^Fig')
    for p, path in enumerate(all_paths):
        if (path == '../output/extracted_0/40-REAL/Phase_2_-_Travaux_CF/13-Suivi_marché/03-Réceptions/2017-01/Passy_avancement_fabrication_janvier_2017_rev.0.xlstxt'):
            begin = True
        if (begin and (exp.match(path.split('/')[-1]) == None)):
            left_paths.append(path)
    return left_paths

# SENTENCE TOKENIZATION
def txt_to_sentences(file_path):    
    with open(file_path, "r") as text:
        text = str(text.read())
        try:
            text=textacy.preprocess.normalize_whitespace(text)
            text=textacy.preprocess.fix_bad_unicode(text, normalization=u'NFKC')
            text=textacy.preprocess.unpack_contractions(text)
            sents = [['{}\t{}\t{}\n'.format(w.text, french_stemmer.stem(w.text), w.pos_) for w in sent if w.pos_ != 'SPACE'] for sent in nlp(text).sents]
        except ValueError as e:
            print('ERROR: {}'.format(file_path))
            print('{}\n'.format(e))
            sents = []
    return sents 

# WRITE TOKENIZED TEXTS IN NEW FILES
def tokenize_files(paths, source_root, destination_root):
    for p, path in enumerate(paths):
        sentences = txt_to_sentences(path)
        # CHANGE AND CREATE PATHS
        if not os.path.exists(destination_root):
            os.makedirs(destination_root)
        new_path = path.replace(source_root, destination_root)     
        repertoire = '/'.join(new_path.split('/')[:-1])
        title = new_path.split('/')[-1]
        # CREATE THE NEW ARBORESCENCE
        directory = repertoire.split('/')
        for d in range(1, len(directory)):
            directory[d] = '/'.join([directory[d-1], directory[d]])
            if not os.path.exists(directory[d]):
                os.makedirs(directory[d])
        # WRITE TOKENIZED FILES
        with open('{}/{}'.format(repertoire, title), 'w') as tokenized_txt:
            content = ''
            for s, sent in enumerate(sentences):
                for w, word in enumerate(sent):
                    content += word
                content += '\n'
            tokenized_txt.write(content)
            tokenized_txt.close()
        print('{}/{}'.format(repertoire, title))

tokenize_files(path_generator(sys.argv[1]), sys.argv[1], sys.argv[2])
