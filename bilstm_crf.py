import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import io

torch.manual_seed(1)


# 默认torch.cuda.is_available() == True


def preprocess(file_path):
    """
    :param file_path: path for corpus in CoNLL-format
    :return: word_set, tag_set - list of sentences, splitted into words, tags
    """
    data = []
    word = []
    tag = []
    char_seq = []  # 每个句子的每个词的字符表示
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0 or len(word) >= max_seq_len:
                if len(word) > 0:
                    word = word + [PAD] * (max_seq_len - len(word))
                    tag = tag + [PAD] * (max_seq_len - len(tag))
                    char_seq = char_seq + [[char_to_ix[PAD]] * max_word_len] * (max_seq_len - len(char_seq))
                    data.append((word, tag, char_seq))
                    word = []
                    tag = []
                    char_seq = []
                continue
            pair = line.split('\t')

            # prepare char dictionary
            w = pair[0]
            if w.isdigit():
                w = "0"
            char = []  # character index representations
            for c in w:
                if c not in char_to_ix:
                    char_to_ix[c] = len(char_to_ix)
                char.append(char_to_ix[c])
            char = char + [char_to_ix[PAD]] * (max_word_len - len(char))
            char_seq.append(char)

            word.append(w.lower())
            tag.append(pair[1])

    if len(word) > 0:
        word = word + [PAD] * (max_seq_len - len(word))
        tag = tag + [PAD] * (max_seq_len - len(tag))
        char_seq = char_seq + [[char_to_ix[PAD]] * max_word_len] * (max_seq_len - len(char_seq))
        data.append((word, tag, char_seq))

    return data  # [([word], [tag], [char]), ...]


def get_embeddings(word_embeddings_path, embedding_size):
    word2idx = {}
    idx2word = {}
    word_embeddings = []

    word2idx[PAD] = len(word2idx)
    idx2word[0] = PAD
    word_embeddings.append(np.zeros(embedding_size))

    word2idx[UNK] = len(word2idx)  # roughly 3% of the training set
    idx2word[1] = UNK
    word_embeddings.append(np.random.uniform(-0.25, 0.25, embedding_size))

    with io.open(word_embeddings_path, 'r', encoding="utf-8") as f_em:
        for line in f_em:
            split = line.strip().split(" ")
            if len(split) <= 2:
                continue
            if len(split) - 1 != embedding_size:
                continue
            word_embeddings.append(np.asarray(split[1:], dtype='float32'))
            word2idx[split[0]] = len(word2idx)
            idx2word[len(word2idx) - 1] = split[0]

    return word_embeddings, word2idx, idx2word


def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    idxs = []
    for w in seq:
        try:
            idxs.append(to_ix[w])
        except:
            idxs.append(to_ix[UNK])
    return torch.tensor(idxs, dtype=torch.long)


class char_Embedding(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(len(char_to_ix), embed_dim, padding_idx=char_to_ix[PAD])
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, input):
        # input: [word_len, batch * seq_len]
        word_len, bs = input.size()
        # 求mask，按词的长度降序排序
        mask = torch.ne(input, torch.tensor(char_to_ix[PAD]).cuda()).float()  # [word_len, batch * seq_len]
        length = mask.sum(0).long()
        max_length, _ = torch.max(length, 0)
        max_length = max_length.item()

        _, sort_idx = length.sort(0, descending=True)
        _, unsort_idx = sort_idx.sort(0)
        x = input[:max_length, sort_idx]
        mask = mask[:max_length, sort_idx]

        length = mask.sum(0).long()
        index = torch.gt(length, torch.tensor(0).cuda()).long().sum(0).item()
        x = x[:, :index]
        mask = mask[:, :index]

        x = self.embedding(x)  # [word_len, index, embed_dim]
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(0).long())
        x, (h, c) = self.bilstm(x)  # h: [num_layers * num_directions, index, hidden_dim]
        h = torch.cat((h[0], h[1]), -1)  # [index, hidden_dim]
        h = torch.cat((h, torch.zeros(bs - index, self.hidden_dim).cuda()), 0)
        h = h[unsort_idx, :]
        return h  # [batch * seq_len, hidden_dim]


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.char_embeds = char_Embedding(CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM)
        self.word_embeds = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings), freeze=False)

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(embedding_dim + CHAR_HIDDEN_DIM, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)  # input: [batch, seq_len, embed_dim]

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.linear = nn.Linear(self.tagset_size, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.word_embeds.weight)
        nn.init.xavier_normal_(self.hidden2tag.weight)
        nn.init.xavier_normal_(self.linear.weight)

    def init_hidden(self, batch=1):
        return (torch.randn(2, batch, self.hidden_dim // 2).cuda(),
                torch.randn(2, batch, self.hidden_dim // 2).cuda())  # [num_layers*num_directions, batch, hidden_size]

    def _forward_alg(self, feats):
        # feats: [batch, seq_len, tag_size]
        batch, seq_len, tag_size = feats.size()
        # Do the forward algorithm to compute the partition function
        alpha = torch.full((batch, self.tagset_size), -10000).cuda()
        # START_TAG has all of the score.
        alpha[:, self.tag_to_ix[START_TAG]] = 0

        convert_feats = feats.transpose(0, 1)  # [seq_len, batch, tag_size]

        # Iterate through the sentence
        for feat in convert_feats:  # [batch, tag_size]
            # [batch, next_tag, current_tag]
            # emit_score is the same regardless of current_tag,
            # so we broadcast along current_tag dimension
            emit_score = feat.unsqueeze(-1)
            # alpha is the same regardless of next_tag,
            # so we broadcast along next_tag dimension
            alpha = alpha.unsqueeze(1) + self.transitions + emit_score  # [batch, tag_size, tag_size]
            # log_sum_exp along current_tag dimension to get next_tag alpha
            alpha = torch.logsumexp(alpha, dim=-1)  # [batch, tag_size]
        alpha = alpha + self.transitions[self.tag_to_ix[STOP_TAG]]
        return torch.logsumexp(alpha, dim=-1)  # [batch]

    def _get_lstm_features(self, sentence, char):
        # sentence: [batch, seq_len]
        # char: [batch, seq_len, word_len]
        hidden = self.init_hidden(batch=sentence.size()[0])

        batch, seq_len, word_len = char.size()
        char_t = char.view(-1, word_len).transpose(0, 1)  # [word_len, batch * seq_len]
        char_embed = self.char_embeds(char_t)  # [batch * seq_len, embed_dim]
        char_embed = char_embed.view(batch, seq_len, -1)  # [batch, seq_len, embed_dim]
        word_embed = self.word_embeds(sentence)  # [batch, seq_len, embed_dim]
        embeds = torch.cat((char_embed, word_embed), dim=-1)

        embeds = self.dropout(embeds)

        lstm_out, hidden = self.lstm(embeds, hidden)  # lstm_out: [batch, seq_len, hidden_dim]
        lstm_feats = F.relu(self.hidden2tag(lstm_out))
        lstm_feats = self.linear(lstm_feats)
        return lstm_feats  # [batch, seq_len, tag_size]

    def _score_sentence(self, feats, tags):
        # feats: [batch, seq_len, tag_size]
        # tags: [batch, seq_len]
        batch = feats.size()[0]
        # Gives the score of a provided tag sequence
        score = torch.zeros(batch).cuda()
        var = torch.full((batch, 1), self.tag_to_ix[START_TAG], dtype=torch.long).cuda()

        tags = torch.cat((var, tags), dim=1)  # 拼接START_TAG
        for i in range(feats.size()[1]):
            feat = feats[:, i, :]  # [batch, tag_size]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(batch), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score  # [batch]

    def _viterbi_decode(self, feats):
        # feats: [batch, seq_len, tag_size]
        batch, seq_len, tag_size = feats.size()
        pointers = []

        # Initialize the viterbi variables in log space
        scores = torch.full((batch, tag_size), -10000).cuda()
        scores[:, self.tag_to_ix[START_TAG]] = 0

        # scores at step i holds the viterbi variables for step i-1
        convert_feats = feats.permute(1, 0, 2)
        for feat in convert_feats:
            # feat: [batch, tag_size]
            # [batch, next_tag, current_tag]
            scores = scores.unsqueeze(1) + self.transitions  # [batch, tag_size, tag_size]
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores, pointer = torch.max(scores, -1)
            scores = scores + feat  # [batch, tag_size]
            pointers.append(pointer)
        pointers = torch.stack(pointers, 0).permute(1, 0, 2)  # [batch, seq_len, tag_size]
        scores = scores + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_score, best_tag = torch.max(scores, -1)  # [batch]

        # Follow the back poinnters to decode the best path
        best_path = best_tag.unsqueeze(-1).tolist()
        for i in range(batch):
            best_tag_i = best_tag[i]
            for ptr_t in reversed(pointers[i]):
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path[i].pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path[i].reverse()
        return best_path

    def neg_log_likelihood(self, sentence, tags, char):
        # 损失函数
        batch = sentence.size()[0]
        feats = self._get_lstm_features(sentence, char)  # BiLSTM+Linear层的输出
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score) / batch

    def forward(self, sentence, char):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, char)

        # Find the best path, given the features.
        tag_seq_list = self._viterbi_decode(lstm_feats)
        return tag_seq_list


def train():
    word_set = []
    tag_set = []
    char_set = []
    for sentence, tags, char_seq in training_data:
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_to_ix[word])
            except:
                sequence.append(word_to_ix[UNK])
        word_set.append(sequence)
        tag_set.append([tag_to_ix[tag] for tag in tags])
        char_set.append(char_seq)

    print("train_size:", len(word_set))
    print("char_size:", len(char_set))

    train_set = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long), torch.tensor(tag_set, dtype=torch.long),
                                   torch.tensor(char_set, dtype=torch.long))
    train_loader = Data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT).cuda()
    # model = torch.load("model/bilstm_crf.pkl").cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    steps = len(word_set) // BATCH_SIZE
    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(train_loader):
            model.zero_grad()

            sentence, tags, char_seq = batch  # char_seq: [batch, seq_len, word_len]

            loss = model.neg_log_likelihood(sentence.cuda(), tags.cuda(),
                                            char_seq.cuda())

            loss.backward()
            optimizer.step()

            print("epoch", epoch, ":", step, "/", steps, "loss:", loss.item())

    torch.save(model, "model/bilstm_crf.pkl")


def tag_convert(tag):  # For evaluation using conlleval.perl, which doesn't support the following.
    if tag == "OUT" or tag == START_TAG or tag == STOP_TAG or tag == PAD:
        return "O"
    else:
        return tag


def test():
    word_set = []
    tag_set = []
    char_set = []
    for sentence, tags, char_seq in test_data:
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_to_ix[word])
            except:
                sequence.append(word_to_ix[UNK])
        word_set.append(sequence)
        tag_set.append([tag_to_ix[tag] for tag in tags])
        char_set.append(char_seq)

    print("test_size:", len(word_set))

    test_set = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long), torch.tensor(tag_set, dtype=torch.long),
                                  torch.tensor(char_set, dtype=torch.long))
    test_loader = Data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE
    )

    model = torch.load("model/bilstm_crf.pkl").cuda()
    # model.eval()
    predict = []
    for batch in test_loader:
        sentence, tags, char_seq = batch
        ans = model(sentence.cuda(), char_seq.cuda())  # [[], [], ...]
        for i in range(len(sentence)):
            predict.append((sentence[i], tags[i], ans[i]))  # Each tuple is a sentence, its tags and its answers.

    with open("results.txt", "w") as f:
        for sentence, tags, ans in predict:
            for i in range(max_seq_len):
                if sentence[i] == word_to_ix[PAD]:
                    break
                else:
                    f.write(ix_to_word[sentence[i].item()] + ' ' \
                            + tag_convert(ix_to_tag[tags[i].item()]) + ' ' \
                            + tag_convert(ix_to_tag[ans[i]]) + '\n')
                    # Note that sentence and tags are tensors, but ans are not tensors.
            f.write('\n')


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD = "<PAD>"
    UNK = "<UNK>"
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 200
    CHAR_EMBEDDING_DIM = 100
    CHAR_HIDDEN_DIM = 50
    DROPOUT = 0.5
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    max_seq_len = 50
    max_word_len = 25

    word_embeddings_path = "../data/glove.6B.100d.txt"
    embedding_size = 100
    
    char_to_ix = {PAD: 0}

    training_data = preprocess("../data/conll.train")
    test_data = preprocess("../data/conll.test")

    word_embeddings, word_to_ix, ix_to_word = get_embeddings(word_embeddings_path, embedding_size)

    tag_to_ix = {PAD: 0, START_TAG: 1, STOP_TAG: 2}
    ix_to_tag = {0: PAD, 1: START_TAG, 2: STOP_TAG}
    for sentence, tags, _ in training_data + test_data:
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
                ix_to_tag[len(ix_to_tag)] = tag
        for word in sentence:
            if word.isdigit():
                word = "0"
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)  # roughly 3% of the training set
                ix_to_word[len(ix_to_word)] = word
                word_embeddings.append(np.random.uniform(-0.25, 0.25, embedding_size))

    print(char_to_ix)
    print(tag_to_ix)
    print(ix_to_tag)

    train()
    test()