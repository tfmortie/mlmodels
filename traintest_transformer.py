"""
Code for training and testing of transformer.

Thomas Mortier
March 2022
"""
import argparse
import torch
import time
import io
import os
import pickle
import torch.nn as nn
import torch.optim as optim

from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from mlmodels.transformer import Transformer, MultiHeadAttention

def save_vocab(vs, vt, svo, tvo):
    print("Saving vocabs to {0} and {1}...".format(svo,tvo))
    file_s = open(svo, 'wb')
    file_t = open(tvo, 'wb')
    pickle.dump(vs, file_s)
    pickle.dump(vt, file_t)
    file_s.close()
    file_t.close()

def load_vocab(svo, tvo):
    print("Loading vocabs {0} and {1}...".format(svo,tvo))
    file_s = open(svo, 'rb')
    file_t = open(tvo, 'rb')
    vs = pickle.load(file_s)
    vt = pickle.load(file_t)
    file_s.close()
    file_t.close()
    # set default token for both vocabs
    vs.set_default_index(vs['<unk>'])
    vt.set_default_index(vt['<unk>'])
    
    return vs, vt

def get_vocab(svo, tvo, f, sl, tl, mt):
    # first check if vocab already exists
    if os.path.exists(svo) and os.path.exists(tvo):
        vocab_s, vocab_t = load_vocab(svo, tvo)
    else: 
        vocab_s = build_vocab_from_iterator(yield_tokens(f+sl, sl), specials=["<unk>","<bos>","<eos>","<pad>"], max_tokens=mt)
        vocab_t = build_vocab_from_iterator(yield_tokens(f+tl, tl), specials=["<unk>","<bos>","<eos>","<pad>"], max_tokens=mt)
        # set default token for both vocabs
        vocab_s.set_default_index(vocab_s['<unk>'])
        vocab_t.set_default_index(vocab_t['<unk>'])
        # and save
        save_vocab(vocab_s, vocab_t, svo, tvo)

    return vocab_s, vocab_t

def yield_tokens(file_path, lang):
    tokenizer = get_tokenizer("spacy",lang)
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield tokenizer(line.strip())

def data_loader(vs, vt, f, sl, tl, trs, b):
    iter_s = yield_tokens(f+sl, sl)
    iter_t = yield_tokens(f+tl, tl)
    batch = []
    for i,(s,t) in enumerate(zip(iter_s,iter_t)):
        if trs != -1:
            if trs < i:
                break
        # process batch
        batch.append((s,t))
        if ((i+1) % b == 0):
            yield process_batch(vs, vt, batch)
            batch = []

def process_batch(vs, vt, batch):
    # run over batch and get max token length for s and t
    max_s, max_t = 0, 0 
    for b in batch:
        s, t = b
        if len(s) > max_s:
            max_s = len(s)
        if len(t) > max_t:
            max_t = len(t)
    max_s+=2
    max_t+=3
    # now process batch
    batch_s, batch_t = [], []
    for b in batch:
        s, t = b
        # get encoding of s and pad
        s = [vs["<bos>"]]+[vs[si] for si in s]+[vs["<eos>"]]
        s = s+[vs["<pad>"]]*(max_s-len(s))
        t = [vt["<bos>"]]+[vt[ti] for ti in t]+[vt["<eos>"]]
        t = t+[vt["<pad>"]]*(max_t-len(t))
        batch_s.append(torch.tensor(s,dtype=torch.long).view(1,-1))
        batch_t.append(torch.tensor(t,dtype=torch.long).view(1,-1))
    
    return torch.cat(batch_s,dim=0), torch.cat(batch_t,dim=0)

def get_pad_mask(src, vocab):
    mask = (src == vocab["<pad>"])

    return mask

def traintransformer(args):
    print(f'{args=}')
    print("Reading in data {0}...".format(args.f))
    vocab_s, vocab_t = get_vocab(args.svo, args.tvo, args.f, args.sl, args.tl, args.mt)
    # create CUDA device
    if args.dev == -2:
        args.device = torch.device("cpu")
    elif args.dev == -1:
        args.device = torch.device("cuda")
    elif args.dev >= 0:
        args.device = torch.device("cuda:{0}".format(args.dev))
    else:
        print("Wrong cuda device number = {0}".format(args.dev))
        exit(1)
    # create model
    model = Transformer(len(vocab_s), len(vocab_t), args)
    model = model.to(args.device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{pytorch_total_params=}')
    optimizer = optim.Adam(model.parameters(), lr=args.l, betas=(0.9,0.98), eps=1e-09)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_t["<pad>"])
    cntr, start_time = 0, time.time()
    for e in range(args.e):
        dl = data_loader(vocab_s, vocab_t, args.f, args.sl, args.tl, args.trs, args.b)
        train_loss = 0.0
        for i, (batch_s, batch_t) in enumerate(dl):
            batch_s, batch_t = batch_s.to(args.device), batch_t.to(args.device)
            pad_mask_s, pad_mask_t = get_pad_mask(batch_s, vocab_s), get_pad_mask(batch_t[:,:-1], vocab_t)
            optimizer.zero_grad()
            # forward pass
            o = model(batch_s, batch_t[:,:-1], pad_mask_s, pad_mask_t)
            # get loss
            loss = criterion(o.view(-1,len(vocab_t)), batch_t[:,1:].reshape(-1))
            # calculate gradients
            loss.backward()
            train_loss += loss.item()
            # backprop
            optimizer.step()
            #tensor_to_preds(o, vocab_t)
            if i % args.i == args.i-1:
                stop_time = time.time()
                processed_time = stop_time-start_time
                print("Epoch {0} training loss = {1} in {2}s".format(e,train_loss/args.i,processed_time))
                train_loss = 0
                start_time = time.time()
                cntr+=1
            if args.it != -1:
                if cntr == args.it:
                    break
        if args.it != -1:
            if cntr == args.it:
                break
    # save model
    model.save()

def testtransformer(args):
    print(f'{args=}')
    print("Reading in data {0}...".format(args.f))
    vocab_s, vocab_t = get_vocab(args.svo, args.tvo, args.f, args.sl, args.tl, args.mt)
    # create CUDA device
    if args.dev == -2:
        args.device = torch.device("cpu")
    elif args.dev == -1:
        args.device = torch.device("cuda")
    elif args.dev >= 0:
        args.device = torch.device("cuda:{0}".format(args.dev))
    else:
        print("Wrong cuda device number = {0}".format(args.dev))
        exit(1)
    # load model
    print("Load model...")
    model = torch.load(args.mo)
    model = model.to(args.device)
    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{pytorch_total_params=}')
    # sentences that need to be translated
    sentences = [
        "It records behaviour.",
        "I am a person.",
        "He was the first to come up with the specific mechanism, that's natural selection.", 
        "More importantly I think, it records behavior.", 
        "How did everything fit together?"]
    print(f'{sentences=}')
    # tokenizer for our sentences
    tokenizer_src = get_tokenizer("spacy",args.sl)
    # process each sentence above
    preds = []
    for s in sentences:
        s_tokenized = tokenizer_src(s)
        s_tensor = torch.tensor([vocab_s["<bos>"]]+[vocab_s[s] for s in s_tokenized]+[vocab_s["<eos>"]],dtype=torch.long).view(1,-1).to(args.device)
        t_tensor = torch.tensor(vocab_t["<bos>"],dtype=torch.long).view(1,-1).to(args.device)
        pred = []
        while t_tensor[0,-1].item() != vocab_t["<eos>"] and len(pred)<=args.ml:
            with torch.no_grad():
                o = model(s_tensor,t_tensor,None,None,False)
                # get token
                pred_token_ind = torch.argmax(nn.functional.softmax(o,dim=-1),dim=-1)[0,-1]
                t_tensor = torch.cat([t_tensor, torch.tensor([pred_token_ind],dtype=torch.long).view(1,-1).cuda()],dim=1)
                pred.append(vocab_t.get_itos()[pred_token_ind])
        preds.append(" ".join(pred))
    print("")
    print(f'{preds=}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Transformer for machine translation.")
    # general model arguments
    parser.add_argument('--train',dest="m",action='store_true')
    parser.add_argument('--test',dest="m",action='store_false')
    parser.add_argument("-e",dest="e",type=int,default=100,help="Number of epochs.")
    parser.add_argument("-i",dest="i",type=int,default=100,help="Number of iterations to print training loss")
    parser.add_argument("-it",dest="it",type=int,default=-1,help="Number of evaluations after which the model should stop training.")
    parser.add_argument("-b",dest="b",type=int,default=128,help="Batch size.")
    parser.add_argument("-ml",dest="ml",type=int,default=5000,help="Max. length of target and source sequences.")
    parser.add_argument("-dev",dest="dev",type=int,default=-1,help="Device for model training and inference (-2:CPU, -1:default GPU, >=0:GPUx).")
    parser.add_argument("-mo",dest="mo",type=str,default="./tmodel.pt",help="Path to file where model is stored after training or loaded for inference.")
    # transformer arguments
    parser.add_argument("-dk",dest="dk",type=int,default=64,help="Dimensionality of keys and queries for self-attention module.")
    parser.add_argument("-dv",dest="dv",type=int,default=64,help="Dimensionality of values for self-attention module.")
    parser.add_argument("-dm",dest="dm",type=int,default=512,help="Dimensionality of sub-layers and embedding layers.")
    parser.add_argument("-dff",dest="dff",type=int,default=512,help="Dimensionality of inner-layers.")
    parser.add_argument("-nh",dest="nh",type=int,default=8,help="Number of heads in self-attention module.")
    parser.add_argument("-ns",dest="ns",type=int,default=3,help="Stack size for encoder and decoder in transformer.")
    parser.add_argument("-d",dest="d",type=float,default=0.1,help="Dropout rate")
    # optimizer arguments
    parser.add_argument("-l",dest="l",type=float,default=0.0001,help="Learning rate for Adam optimizer.")
    # data arguments
    parser.add_argument("-f",dest="f",type=str,default="/data/home/thomasm/data/opensubtitles/OpenSubtitles.en-nl.",help="Prefix of path that is used for reading in source and target data.")
    parser.add_argument("-sl",dest="sl",type=str,default="en",help="Source language.")
    parser.add_argument("-tl",dest="tl",type=str,default="nl",help="Target language.")
    parser.add_argument("-svo",dest="svo",type=str,default="./vocab_s.pkl",help="Path to file where vocab for source is stored.")
    parser.add_argument("-tvo",dest="tvo",type=str,default="./vocab_t.pkl",help="Path to file where vocab for target is stored.")
    parser.add_argument("-mt",dest="mt",type=int,default=20000,help="If provided, creates the vocab from the mt most frequent tokens.")
    parser.add_argument("-trs",dest="trs",type=int,default=-1,help="Size of training data")
    args = parser.parse_args()
    if args.m:
        # train
        traintransformer(args) 
    # test
    testtransformer(args) 
