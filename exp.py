import torch

def preprocess(file_path):
    """
    :param file_path: path for corpus in CoNLL-format
    :return: word_set, tag_set - list of sentences, splitted into words, tags
    """
    training_data = []
    word = []
    tag = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0 or len(word) >= max_seq_len:
                if len(word) > 0:
                    word = word + [PAD_TAG] * (max_seq_len - len(word))
                    tag = tag + [PAD_TAG] * (max_seq_len - len(tag))
                    training_data.append((word, tag))
                    word = []
                    tag = []
                continue
            pair = line.split('\t')
            word.append(pair[0])
            tag.append(pair[1])

    if len(word) > 0:
        word = word + [PAD_TAG] * (max_seq_len - len(word))
        tag = tag + [PAD_TAG] * (max_seq_len - len(tag))
        training_data.append((word, tag))

    return training_data  # [([word], [tag]), ...]


START_TAG = '<START>'
STOP_TAG = '<STOP>'
PAD_TAG = '<PAD>'
max_seq_len = 10

training_data = preprocess("data/conll.train")

word_to_ix = {}
word_to_ix[PAD_TAG] = 0
tag_to_ix = {START_TAG: 0, STOP_TAG: 1, PAD_TAG: 2}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

word_set = []
tag_set = []
for sentence, tags in training_data:
    word_set.append([word_to_ix[word] for word in sentence])
    tag_set.append([tag_to_ix[tag] for tag in tags])
