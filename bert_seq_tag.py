import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer,BertModel
import torch.nn as nn
import torch.optim as optim


seed_val = 5
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=False)
VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(Dataset):


    def __init__(self, fpath):
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        x, y = [], [] # list of ids
        valid_ids = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            valid_id = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            valid_ids.extend(valid_id)
            y.extend(yy)

        assert len(x)==len(y)==len(valid_ids), f"len(x)={len(x)}, len(y)={len(y)}, len(valid_ids)={len(valid_ids)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, valid_ids, tags, y, seqlen


class ner_head(nn.Module):
    def __init__(self,  vocab_size=None, device='cpu'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        

    def forward(self, x, y, ):
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            enc = self.bert(x)[0]
           
        else:
            self.bert.eval()
            with torch.no_grad():
                enc= self.bert(x)[0]

       
        enc, _ = self.rnn(enc)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


def pad(batch):
    #padding to the longest sample
    batch_data = lambda x: [sample[x] for sample in batch]
    words = batch_data(0)
    valid_ids = batch_data(2)
    tags = batch_data(3)
    seqlens = batch_data(-1)
    maxlen = np.array(seqlens).max()

    pad_to_max = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = pad_to_max(1, maxlen)
    y = pad_to_max(-2, maxlen)

    return words, torch.LongTensor(x), valid_ids, tags, torch.LongTensor(y), seqlens

def train(model, iterator, optimizer, loss_fn):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, valid_ids, tags, y, seqlens = batch
        _y = y 
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)
        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)
        loss = loss_fn(logits, y)
        loss.backward()

        optimizer.step()

        if i%100==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator):
    model.eval()

    Words, Valid_ids, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, valid_ids, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Valid_ids.extend(valid_ids)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())


        
    res_true=[]
    res_pred=[]
    for valid_ids, tags, y_hat in zip( Valid_ids, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(valid_ids, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        for t, p in zip(tags.split()[1:-1], preds[1:-1]):
            res_true.append(t)
            res_pred.append(p)
            

    ## calc metric
    y_true =  np.array([tag2idx[t] for t in res_true])
    y_pred =  np.array([tag2idx[p] for p in res_pred])

    num_predicted = len(y_pred[y_pred>1])
    num_correctly_predicted = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_true = len(y_true[y_true>1])
    print()
    print(f"Number of tags predicted:{num_predicted}")
    print(f"Number of actual tags:{num_true}")
    print(f"Number of tags correctly predicted:{num_correctly_predicted}")
   
    precision = num_correctly_predicted / num_predicted
   
    recall = num_correctly_predicted / num_true
    
    f1 = 2*precision*recall / (precision + recall)
    
    print()
    print("Precision = %.2f"%precision)
    print("Recall = %.2f"%recall)
    print("F1 = %.2f"%f1)
    print()
    return precision, recall, f1

if __name__=="__main__":

    batch_size=16
    n_epochs=3
    trainset="data/train.txt"
    validset="data/dev.txt"
    testset="data/test.txt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ner_head(len(VOCAB), device).cuda()

    train_dataset = NerDataset(trainset)
    eval_dataset = NerDataset(validset)
    test_dataset = NerDataset(testset)

    train_iter = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = DataLoader(dataset=eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    test_iter = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, n_epochs+1):
        print(f"=========Training epoch: {epoch} =========")
        train(model, train_iter, optimizer, loss_fn)
        print()
        print(f"=========Evaluation at epoch={epoch}=========")
        precision, recall, f1 = eval(model, eval_iter)
    print("===========Performance on test set: ============")
    eval(model, test_iter)
