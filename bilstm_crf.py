import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

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
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0 or len(word) >= max_seq_len:
                if len(word) > 0:
                    word = word + [PAD_TAG] * (max_seq_len - len(word))
                    tag = tag + [PAD_TAG] * (max_seq_len - len(tag))
                    data.append((word, tag))
                    word = []
                    tag = []
                continue
            pair = line.split('\t')
            word.append(pair[0])
            tag.append(pair[1])

    if len(word) > 0:
        word = word + [PAD_TAG] * (max_seq_len - len(word))
        tag = tag + [PAD_TAG] * (max_seq_len - len(tag))
        data.append((word, tag))

    return data  # [([word], [tag]), ...]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=0.5)  # input: [batch, seq_len,
        # embed_dim]

        self.dropout = nn.Dropout(0.5)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

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
        nn.init.xavier_normal_(self.word_embeds.weight)
        nn.init.xavier_normal_(self.hidden2tag.weight)

    def init_hidden(self, batch=1):
        return (torch.randn(2, batch, self.hidden_dim // 2).cuda(),
                torch.randn(2, batch, self.hidden_dim // 2).cuda())  # [num_layers*num_directions, batch, hidden_size]

    def _forward_alg(self, feats, mask):
        # feats: [batch, seq_len, tag_size]
        # mask: [seq_len, batch]
        batch, seq_len, tag_size = feats.size()
        # Do the forward algorithm to compute the partition function
        alpha = torch.full((batch, self.tagset_size), -10000).cuda()
        # START_TAG has all of the score.
        alpha[:, self.tag_to_ix[START_TAG]] = 0

        convert_feats = feats.transpose(0, 1)  # [seq_len, batch, tag_size]

        # Iterate through the sentence
        for t, feat in enumerate(convert_feats):  # [batch, tag_size]
            # [batch, next_tag, current_tag]
            # emit_score is the same regardless of current_tag,
            # so we broadcast along current_tag dimension
            emit_score = feat.unsqueeze(-1)
            # alpha is the same regardless of next_tag,
            # so we broadcast along next_tag dimension
            alpha_t = alpha.unsqueeze(1) + self.transitions + emit_score  # [batch, tag_size, tag_size]

            mask_t = mask[t].unsqueeze(-1)
            # log_sum_exp along current_tag dimension to get next_tag alpha
            alpha = torch.logsumexp(alpha_t, dim=-1) * mask_t + alpha * (1 - mask_t)  # [batch, tag_size]

        alpha = alpha + self.transitions[self.tag_to_ix[STOP_TAG]]
        return torch.logsumexp(alpha, dim=-1)  # [batch]

    def _get_lstm_features(self, sentence, mask):
        # sentence: [batch, seq_len]
        # mask: [seq_len, batch]
        hidden = self.init_hidden(batch=sentence.size()[0])
        embeds = self.word_embeds(sentence)  # [batch, seq_len, embed_dim]
        embeds = self.dropout(embeds)
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, mask.sum(0).long(), batch_first=True)
        lstm_out, hidden = self.lstm(embeds, hidden)  # lstm_out: [batch, seq_len, hidden_dim]
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out * mask.transpose(0, 1).unsqueeze(-1)
        lstm_feats = self.hidden2tag(lstm_out) * mask.transpose(0, 1).unsqueeze(-1)
        return lstm_feats  # [batch, seq_len, tag_size]

    def _score_sentence(self, feats, tags, mask):
        # feats: [batch, seq_len, tag_size]
        # tags: [batch, seq_len]
        # mask: [seq_len, batch]
        batch = feats.size()[0]
        # Gives the score of a provided tag sequence
        score = torch.zeros(batch).cuda()
        var = torch.full((batch, 1), self.tag_to_ix[START_TAG], dtype=torch.long).cuda()

        tags = torch.cat((var, tags), dim=1)  # 拼接START_TAG
        for i in range(feats.size()[1]):
            feat = feats[:, i, :]  # [batch, tag_size]
            score = score + \
                    (self.transitions[tags[:, i + 1], tags[:, i]] + feat[:, tags[:, i + 1]]) * mask[i]
        terminal_var = torch.stack([self.transitions[tag_to_ix[STOP_TAG], tag[mask[:, b].sum().long()]] for b,
                                                                                                                 tag
                                         in enumerate(tags)])
        score = score + terminal_var
        return score  # [batch]

    def _viterbi_decode(self, feats, mask):
        # feats: [batch, seq_len, tag_size]
        # mask: [seq_len, batch]
        batch, seq_len, tag_size = feats.size()
        pointers = []

        # Initialize the viterbi variables in log space
        scores = torch.full((batch, tag_size), -10000).cuda()
        scores[:, self.tag_to_ix[START_TAG]] = 0

        # scores at step i holds the viterbi variables for step i-1
        convert_feats = feats.permute(1, 0, 2)
        for t, feat in enumerate(convert_feats):
            # feat: [batch, tag_size]
            # [batch, next_tag, current_tag]
            scores_t = scores.unsqueeze(1) + self.transitions # [batch, tag_size, tag_size]
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)
            scores_t = scores_t + feat  # [batch, tag_size]
            pointers.append(pointer)

            mask_t = mask[t].unsqueeze(-1)
            scores = scores_t * mask_t + scores * (1 - mask_t)

        pointers = torch.stack(pointers, 0)  # [batch, seq_len, tag_size]
        scores = scores + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_score, best_tag = torch.max(scores, -1)  # [batch]

        # Follow the back poinnters to decode the best path
        best_path = best_tag.unsqueeze(-1).tolist()
        for i in range(batch):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path[i].pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path[i].reverse()
        return best_path

    def neg_log_likelihood(self, sentence, tags, mask):
        # 损失函数
        batch = sentence.size()[0]
        feats = self._get_lstm_features(sentence, mask)  # BiLSTM+Linear层的输出
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence, mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, mask)

        # Find the best path, given the features.
        tag_seq_list = self._viterbi_decode(lstm_feats, mask)
        return tag_seq_list


def train():
    word_set = []
    tag_set = []
    for sentence, tags in training_data:
        word_set.append([word_to_ix[word] for word in sentence])
        tag_set.append([tag_to_ix[tag] for tag in tags])

    print("train_size:", len(word_set))

    train_set = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long), torch.tensor(tag_set, dtype=torch.long))
    train_loader = Data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    '''
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[1][0], word_to_ix).view(1, -1).cuda()
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[1][1]], dtype=torch.long)
        print(precheck_tags)
        print(model(precheck_sent))
    '''

    steps = len(word_set) // BATCH_SIZE
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            model.zero_grad()

            sentence, tags = batch
            mask = torch.ne(sentence, torch.tensor(tag_to_ix[PAD_TAG])).float().cuda()
            mask = mask.transpose(-1, -2)

            loss = model.neg_log_likelihood(sentence.clone().detach().cuda(), tags.clone().detach().cuda(), mask)

            loss.backward()
            optimizer.step()

            print("epoch", epoch, ":", step, "/", steps, "loss:", loss.item())

    torch.save(model, "model/bilstm_crf.pkl")

    '''
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[1][0], word_to_ix).view(1, -1).cuda()
        print(model(precheck_sent))
    '''


def test():
    word_set = []
    tag_set = []
    for sentence, tags in test_data:
        word_set.append([word_to_ix[word] for word in sentence])
        tag_set.append([tag_to_ix[tag] for tag in tags])

    print("test_size:", len(word_set))

    test_set = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long), torch.tensor(tag_set, dtype=torch.long))
    test_loader = Data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE
    )

    model = torch.load("model/bilstm_crf.pkl").cuda()
    for batch in test_loader:
        sentence, tags = batch
        ans = model(sentence.clone().detach().cuda())  # [[], [], ...]
        # TODO: batch evaluate


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 100
    BATCH_SIZE = 32
    max_seq_len = 100
    num_epochs = 25

    training_data = preprocess("data/conll.train")
    test_data = preprocess("data/conll.test")

    word_to_ix = {PAD_TAG: 0}
    tag_to_ix = {START_TAG: 0, STOP_TAG: 1, PAD_TAG: 2}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    for sentence, _ in test_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    ix_to_tag = {}
    for tag in tag_to_ix.keys():
        ix_to_tag[tag_to_ix[tag]] = tag

    print(tag_to_ix)
    print(ix_to_tag)

    train()
    # test()