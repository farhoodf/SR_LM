import stanza
from stanza.utils.conll import CoNLL
from tqdm import tqdm

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
PATH_TO_WIKITEXT = "./wikitext-103-raw/wiki.train.raw"



def remove_at(s):
    return re.sub(r'@ ',r'', re.sub(r' @',r'', s))

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def filter_sentences(sentences):
    filtered = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 150:
            continue
        if not is_english(sentence):
            continue
        sentence = remove_at(sentence)
        filtered.append(sentence)
    return filtered

def get_ud_format(sentence):
    final_text = "# sent_id = something\n"
    final_text += "# text = " + sentence + "\n"
    conll_format = CoNLL.convert_dict(nlp(sentence).to_dict())[0]
    for token in conll_format:
        final_text += '\t'.join(token[:-2]+['_','_']) + "\n"
    return final_text



with open('./wikitext-103-raw/wiki.train.raw','r') as f:
    raw = f.read()

wiki = []
for parag in raw.splitlines():
    if parag != " " and not parag.startswith(' ='):
        for sent in parag.split(' . '):
            if len(sent) > 0:
                wiki.append(sent.strip() + ' .')
filtered = myfilter(wiki)


filtered = filtered[:500000]
to_write = []
for sentence in tqdm(filtered):
    to_write.append(get_ud_format(sentence))
with open(f'./wiki_train_ud.conllu','w') as f:
    f.write('\n'.join(to_write))
    f.write('\n')