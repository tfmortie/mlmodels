"""
Transformer for machine translation (Attention Is All You Need, Vaswani et al. 2018)

Thomas Mortier
March 2022
"""
import torch
import numpy as np

class Transformer(torch.nn.Module):
    """ Represents the main transformer class.
    """
    def __init__(self, voc_s_size, voc_t_size, args):
        super(Transformer, self).__init__()
        self.args = args
        enc_l = [Encoder(args) for _ in range(args.ns)]
        # register the encoder and decoder
        self.encoder = torch.nn.Sequential(*enc_l)
        self.decoder = torch.nn.ModuleList([Decoder(args) for _ in range(args.ns)])
        # register embedding layers for source->embedding and target->embedding
        self.emb_voc_s = torch.nn.Embedding(voc_s_size, args.dm)
        self.emb_voc_t = torch.nn.Embedding(voc_t_size, args.dm)
        # our final layer which predicts target words
        self.emb_dm = torch.nn.Linear(args.dm, voc_t_size)
        # and our positional encoder
        self.pe = PositionalEncoder(args)
        # init all weights
        def init_xavier(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_xavier)

    def forward(self, s, t):
        # get embeddings for source and target sequence
        s = self.emb_voc_s(s)
        t = self.emb_voc_t(t)
        # pass both through positional encoder
        s = self.pe(s)
        t = self.pe(t)
        # pass through encoder network
        e_o = self.encoder(s)
        # and finally pass through decoder
        d_o = t
        for d in self.decoder:
            d_o = d(d_o, e_o, True)
        d_o = self.emb_dm(d_o)

        return d_o

    def save(self):
        print("Saving model to {0}...".format(self.args.mo))
        torch.save(self, self.args.mo)

class PositionalEncoder(torch.nn.Module):
    def __init__(self, args):
        super(PositionalEncoder, self).__init__()
        self.args = args
        pos = torch.arange(0,args.ml).view(-1,1).repeat((1,args.dm)) 
        denum = 10000**((2*torch.arange(0,args.dm).view(1,-1).repeat((args.ml,1)))/args.dm)
        PE = pos/denum
        c = torch.arange(0,args.dm).view(1,-1).repeat((args.ml,1))%2
        s = 1-c
        self.PE = torch.cos(c*PE)+torch.sin(s*PE)
        self.PE = self.PE.to(args.device)
        self.dropout = torch.nn.Dropout(p=args.d)

    def forward(self, x):
        x = x + self.PE[:x.shape[1],:].unsqueeze(0).repeat(x.shape[0],1,1)
        x = self.dropout(x) 

        return x

class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.mha = MultiHeadAttention(args)
        self.norml = torch.nn.LayerNorm((args.dm))
        self.linear = torch.nn.Sequential(
                torch.nn.Linear(args.dm, args.dff),
                torch.nn.ReLU(),
                torch.nn.Linear(args.dff, args.dm)
        )
        self.dropout = torch.nn.Dropout(p=args.d)

    def forward(self, x):
        o1 = self.mha(x)
        o1 = self.dropout(o1)
        # add skip connection
        o1 = o1 + x
        # normalize features of each token
        o1 = self.norml(o1)
        # pass through fully-connected network
        o2 = self.linear(o1)
        o2 = self.dropout(o2)
        # add second skip connection
        o2 = o2 + o1
        # and normalize again
        o2 = self.norml(o2)

        return o2

class Decoder(torch.nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.mha1 = MultiHeadAttention(args)
        self.mha2 = MultiHeadAttention(args)
        self.norml = torch.nn.LayerNorm((args.dm))
        self.linear = torch.nn.Sequential(
                torch.nn.Linear(args.dm, args.dff),
                torch.nn.ReLU(),
                torch.nn.Linear(args.dff, args.dm)
        )
        self.dropout = torch.nn.Dropout(p=args.d)

    def forward(self, x, e, mask):
        o1 = self.mha1(x, None, None, mask)
        o1 = self.dropout(o1)
        # add skip connection
        o1 = o1 + x
        # normalize features of each token
        o1 = self.norml(o1)
        # calculate attention with encoded source 
        o2 = self.mha2(o1,e,e,-1)
        o2 = self.dropout(o2)
        # add second skip connection
        o2 = o2 + o1
        o2 = self.norml(o2)
        o3 = self.linear(o2)
        o3 = self.dropout(o3)
        # add third and final skip connection
        o3 = o3 + o2

        return o3
        
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        # define projection matrices for Q,K and V
        self.w_q = torch.nn.Linear(args.dm,args.dm)
        self.w_k = torch.nn.Linear(args.dm,args.dm)
        self.w_v = torch.nn.Linear(args.dm,args.dm)
        assert (args.dv*args.nh)==args.dm, 'Number of heads must be multiple of dimensionality of K and V!'
        self.linear = torch.nn.Linear(args.dv*args.nh,args.dm)
        self.sldpa = ScaledLinearDotProductAttention(args)

    def forward(self, q, k=None, v=None, mask=False):
        # calculate projections
        if k is None:
            proj_q = self.w_q(q)
            proj_k = self.w_k(q)
            proj_v = self.w_v(q)
        else:
            proj_q = self.w_q(q)
            proj_k = self.w_k(k)
            proj_v = self.w_v(v)
        # get chunks and perform attention
        proj_q_l = torch.chunk(proj_q,self.args.nh,2)
        proj_k_l = torch.chunk(proj_k,self.args.nh,2)
        proj_v_l = torch.chunk(proj_v,self.args.nh,2)
        o_l = []
        for i in range(self.args.nh):
            o_l.append(self.sldpa(proj_q_l[i], proj_k_l[i], proj_v_l[i], mask))
        o = torch.cat(o_l,dim=-1)
        o = self.linear(o)

        return o

class ScaledLinearDotProductAttention(torch.nn.Module):
    def __init__(self, args):
        super(ScaledLinearDotProductAttention, self).__init__()
        self.args = args

    def forward(self, ql, kl, vl, mask=False):
        o = torch.einsum('bij,bjk->bik',ql,torch.einsum('bij->bji',kl))
        o = o/np.sqrt(self.args.dk)
        if mask:
            o = o + torch.triu(torch.zeros_like(o)-torch.inf,diagonal=1).to(o.device) # reduces memory usage
        o = torch.nn.functional.softmax(o,dim=-1)
        o = torch.einsum('bij,bjk->bik',o,vl)

        return o
