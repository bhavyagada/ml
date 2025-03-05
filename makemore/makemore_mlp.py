# https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

import random
from tqdm import trange
import torch
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

# magic numbers (play with them to get better results)
block_size = 4
emb_size = 20
batch_size = 128
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
hidden_size = 500
epochs = 400000

def build_dataset(words):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + ".":
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append
  X, Y = torch.tensor(X), torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

random.shuffle(words)
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# initialize model parameters
C = torch.randn((27, emb_size), requires_grad=True)
W1 = torch.randn((block_size * emb_size, hidden_size), requires_grad=True)
b1 = torch.randn(hidden_size, requires_grad=True)
W2 = torch.randn((hidden_size, 27), requires_grad=True)
b2 = torch.randn(27, requires_grad=True)
parameters = [C, W1, b1, W2, b2]
print("total parameters: ", sum(p.nelement() for p in parameters))

def train(X, Y):
  stepi = []
  lossi = []

  for i in (t := trange(epochs)):
    # minibatch
    ix = torch.randint(0, X.shape[0], (batch_size,))

    # forward pass
    emb = C[X[ix]] # (batch_size, block_size, emb_size)
    h = torch.tanh(emb.view(-1, block_size * emb_size) @ W1 + b1) # (batch_size, 500)
    logits = h @ W2 + b2 # (batch_size, 27)
    loss = F.cross_entropy(logits, Y[ix])

    # backward pass
    for p in parameters: p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.05 if i < 200000 else 0.01
    for p in parameters: p.data += -lr * p.grad

    # track stats
    stepi.append(i)
    lossi.append(loss.log10().item())

    t.set_description(f"loss: {loss.item():.2f}")
    # return stepi, lossi

def evaluate(X, Y):
  # validation set loss
  emb = C[X] # (batch_size, block_size, emb_size)
  h = torch.tanh(emb.view(-1, block_size * emb_size) @ W1 + b1) # (batch_size, 500)
  logits = h @ W2 + b2 # (batch_size, 27)
  loss = F.cross_entropy(logits, Y)
  print(loss.item())

# train and eval
train(Xtr, Ytr)
# plt.plot(stepi, lossi) # plot
evaluate(Xtr, Ytr) # eval train set
evaluate(Xdev, Ydev) # eval validation set
evaluate(Xte, Yte) # eval test set

# sampling
for _ in range(20):
  out = []
  context = [0] * block_size # initialize with ....
  while True:
    emb = C[torch.tensor([context])] # (1, block_size, emb_size)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
    context = context[1:] + [ix]
    out.append(itos[ix])
    if ix == 0: break
  print("".join(out))
