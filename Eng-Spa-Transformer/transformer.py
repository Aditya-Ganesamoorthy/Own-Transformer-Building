import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import pickle
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATA LOADING ================= #

def load_parallel_data(en_path, ta_path, max_len=20):
    with open(en_path, encoding='utf-8') as f1, open(ta_path, encoding='utf-8') as f2:
        en_lines = f1.read().splitlines()
        ta_lines = f2.read().splitlines()

    data = []
    for en, ta in zip(en_lines, ta_lines):
        if len(en.split()) <= max_len and len(ta.split()) <= max_len:
            data.append((en.lower(), ta.lower()))

    return data

dataset = load_parallel_data("english.txt", "spanish.txt")
print(f"Loaded {len(dataset)} sentence pairs.")

# ================= VOCAB ================= #

def build_vocab(sentences):
    vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
    idx = 4
    for sent in sentences:
        for w in sent.split():
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


if os.path.exists("eng_vocab.pkl"):
    with open("eng_vocab.pkl","rb") as f:
        eng_vocab = pickle.load(f)
    with open("spa_vocab.pkl","rb") as f:
        spa_vocab = pickle.load(f)
    print("Loaded saved vocabularies.")
else:
    eng_sentences = [e for e,_ in dataset]
    spa_sentences = [t for _,t in dataset]

    eng_vocab = build_vocab(eng_sentences)
    spa_vocab = build_vocab(spa_sentences)

    with open("eng_vocab.pkl","wb") as f:
        pickle.dump(eng_vocab,f)
    with open("spa_vocab.pkl","wb") as f:
        pickle.dump(spa_vocab,f)

    print("Built and saved vocabularies.")

inv_spa_vocab = {v:k for k,v in spa_vocab.items()}

def encode_sentence(sent, vocab):
    tokens = [vocab.get(w, vocab["<unk>"]) for w in sent.split()]
    return [vocab["<sos>"]] + tokens + [vocab["<eos>"]]

train_data = [(encode_sentence(e,eng_vocab),
               encode_sentence(t,spa_vocab)) for e,t in dataset]

def pad_sequences(batch, pad_idx=0):
    max_len = max(len(x) for x in batch)
    return [x + [pad_idx]*(max_len - len(x)) for x in batch]


# ================= TRANSFORMER ================= #

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=200):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))

    def forward(self,x):
        return x + self.pe[:,:x.size(1)]

def attention(Q,K,V):
    scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(Q.size(-1))
    return torch.matmul(F.softmax(scores,dim=-1),V)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,heads):
        super().__init__()
        self.d_k = d_model//heads
        self.heads = heads
        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)

    def forward(self,Q,K,V):
        bs = Q.size(0)
        Q = self.Wq(Q).view(bs,-1,self.heads,self.d_k).transpose(1,2)
        K = self.Wk(K).view(bs,-1,self.heads,self.d_k).transpose(1,2)
        V = self.Wv(V).view(bs,-1,self.heads,self.d_k).transpose(1,2)
        out = attention(Q,K,V)
        out = out.transpose(1,2).contiguous().view(bs,-1,self.heads*self.d_k)
        return self.fc(out)

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self,d_model,heads,d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,heads)
        self.ff = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self,x):
        x = self.norm1(x + self.attn(x,x,x))
        return self.norm2(x + self.ff(x))

class DecoderLayer(nn.Module):
    def __init__(self,d_model,heads,d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,heads)
        self.cross_attn = MultiHeadAttention(d_model,heads)
        self.ff = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self,x,enc):
        x = self.norm1(x + self.self_attn(x,x,x))
        x = self.norm2(x + self.cross_attn(x,enc,enc))
        return self.norm3(x + self.ff(x))

class TransformerMT(nn.Module):
    def __init__(self,src_vocab,tgt_vocab,
                 d_model=256,heads=8,layers=3,d_ff=512):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab,d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab,d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model,heads,d_ff) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model,heads,d_ff) for _ in range(layers)])
        self.fc = nn.Linear(d_model,tgt_vocab)

    def forward(self,src,tgt):
        src = self.pos(self.src_embed(src))
        tgt = self.pos(self.tgt_embed(tgt))
        for layer in self.encoder:
            src = layer(src)
        for layer in self.decoder:
            tgt = layer(tgt,src)
        return self.fc(tgt)

# ================= TRAIN OR LOAD ================= #

model = TransformerMT(len(eng_vocab),len(spa_vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0003)
criterion = nn.CrossEntropyLoss(ignore_index=0)

if os.path.exists("eng_spa_transformer.pth"):
    model.load_state_dict(torch.load("eng_spa_transformer.pth",map_location=device))
    model.eval()
    print("Loaded saved model.")

else:
    print("Training model...")
    model.train()

    BATCH_SIZE = 32
    EPOCHS = 12
    total_batches = math.ceil(len(train_data) / BATCH_SIZE)

    for epoch in range(EPOCHS):

        start_time = time.time()
        random.shuffle(train_data)
        total_loss = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        for batch_idx, i in enumerate(range(0, len(train_data), BATCH_SIZE)):

            batch = train_data[i:i+BATCH_SIZE]
            src_batch = pad_sequences([x for x,_ in batch])
            tgt_batch = pad_sequences([y for _,y in batch])

            src = torch.tensor(src_batch).to(device)
            tgt = torch.tensor(tgt_batch).to(device)

            optimizer.zero_grad()
            out = model(src, tgt[:, :-1])

            loss = criterion(
                out.reshape(-1, len(spa_vocab)),
                tgt[:, 1:].reshape(-1)
            )

            loss.backward()

            # Gradient clipping (important for Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                progress = (batch_idx / total_batches) * 100
                print(f"Batch {batch_idx}/{total_batches} "
                      f"({progress:.1f}%) "
                      f"Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} Completed")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Time Taken: {epoch_time:.2f} seconds")

    torch.save(model.state_dict(),"eng_spa_transformer.pth")
    print("Model saved successfully.")

# ================= INFERENCE ================= #

def translate(sentence):
    model.eval()
    enc = encode_sentence(sentence.lower(),eng_vocab)
    enc = torch.tensor([enc]).to(device)

    tgt_seq = [spa_vocab["<sos>"]]

    with torch.no_grad():
        for _ in range(30):
            tgt_tensor = torch.tensor([tgt_seq]).to(device)
            out = model(enc, tgt_tensor)
            probs = torch.softmax(out[0, -1] / 0.8, dim=-1)
            next_word = torch.multinomial(probs, 1).item()
            tgt_seq.append(next_word)

            # Stop if end token predicted
            if next_word == spa_vocab["<eos>"]:
                break

            # Stop if last 3 predicted words are same (repetition control)
            if len(tgt_seq) > 3 and len(set(tgt_seq[-3:])) == 1:
                break

    return " ".join(inv_spa_vocab.get(i,"") for i in tgt_seq[1:-1])

while True:
    text = input("Enter English (type exit): ")
    if text=="exit":
        break
    print("Spanish:",translate(text))
