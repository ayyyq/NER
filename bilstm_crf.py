import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

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


# Helper functions
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    # vec: [batch=1, seq_len]
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    # 等同于return torch.log(torch.sum(torch.exp(vec)))


def log_sum_exp_batch(vec):
    # vec: [batch, seq_len]
    batch = vec.size()[0]
    max_score = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score.view(batch, -1).expand(batch, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))  # [batch]


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
                            num_layers=1, bidirectional=True, batch_first=True)  # input: [batch, seq_len, embed_dim]

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

        self.hidden = self.init_hidden()

    def init_hidden(self, batch=1):
        return (torch.randn(2, batch, self.hidden_dim // 2).cuda(),
                torch.randn(2, batch, self.hidden_dim // 2).cuda())  # [num_layers*num_directions, batch, hidden_size]

    def _forward_alg(self, feats):
        # feats: [batch, seq_len, tag_size]
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        convert_feats = feats.permute(1, 0, 2)  # [seq_len, batch, tag_size]

        # Iterate through the sentence
        # 迭代单词w_i
        for feat in convert_feats:  # [batch, tag_size]
            batch = feat.size()[0]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(
                    batch, -1).expand(batch, self.tagset_size)  # BiLSTM的输出矩阵(P)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)  # CRF的转移矩阵(A)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score  # emit_score有broadcast，next_tag_var: [batch,
                # tag_size]
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp_batch(next_tag_var).view(-1, 1))
            forward_var = torch.cat(alphas_t, dim=1)  # [batch, tag_size]
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # transitions有broadcast
        alpha = log_sum_exp_batch(terminal_var)
        return alpha  # [batch]

    def _get_lstm_features(self, sentence):
        # sentence: [batch, seq_len]
        self.hidden = self.init_hidden(batch=sentence.size()[0])
        embeds = self.word_embeds(sentence)  # [batch, seq_len, embed_dim]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # lstm_out: [batch, seq_len, hidden_dim]
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats  # [batch, seq_len, tag_size]

    def _score_sentence(self, feats, tags):
        # feats: [batch, seq_len, tag_size]
        # tags: [batch, tag_size]
        batch = feats.size()[0]
        # Gives the score of a provided tag sequence
        score = torch.zeros(batch).cuda()
        var = torch.full((batch, 1), self.tag_to_ix[START_TAG], dtype=torch.long).cuda()

        tags = torch.cat((var, tags), dim=1)  # 拼接START_TAG
        for i in range(feats.size()[1]):
            feat = feats[:, i, :]
            score = score + \
                self.transitions[tags[:, i + 1], tags[:, i]] + feat[:, tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        convert_feats = feats.squeeze(0)
        for feat in convert_feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_parallel(self, feats_list):
        # 输出预测序列的路径
        path_list = []
        for feats in feats_list:
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            path_list.append(best_path)
        return path_list

    def neg_log_likelihood(self, sentence, tags):
        # 损失函数
        batch = sentence.size()[0]
        feats = self._get_lstm_features(sentence)  # BiLSTM+Linear层的输出
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag = self._viterbi_decode(lstm_feats)
        return score, tag

    def predict(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        tag_seq_list = self._viterbi_decode_parallel(lstm_feats)
        return tag_seq_list


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 100
    BATCH_SIZE = 64
    max_seq_len = 100
    num_epochs = 15

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

    print("word_set:", len(word_set))
    print("tag_size:", len(tag_to_ix))

    train_set = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long), torch.tensor(tag_set, dtype=torch.long))
    train_loader = Data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[1][0], word_to_ix).view(1, -1).cuda()
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[1][1]], dtype=torch.long)
        print(precheck_tags)
        print(model(precheck_sent))

    steps = len(word_set) // BATCH_SIZE
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            model.zero_grad()

            sentence, tags = batch

            loss = model.neg_log_likelihood(sentence.clone().detach().cuda(), tags.clone().detach().cuda())

            loss.backward()
            optimizer.step()

            print("epoch", epoch, ":", step, "/", steps, "loss:", loss.item())

    torch.save(model, "model/bilstm_crf.pkl")

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[1][0], word_to_ix).view(1, -1).cuda()
        print(model(precheck_sent))