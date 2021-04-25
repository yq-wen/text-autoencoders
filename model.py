import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noise import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_h, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)
        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, is_train=False):
        _input = noisy(self.vocab, input, *self.args.noise) if is_train else input
        mu, logvar = self.encode(_input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}

class DVAE(VAE):
    """Disentangled Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def autoenc(self, x1, x2, y1, y2, is_train=False):
        mu_x1, logvar_x1, z_x1, logits_x1_orig = self(x1, is_train)
        mu_x2, logvar_x2, z_x2, logits_x2_orig = self(x2, is_train)

        split = self.args.dim_z // 2

        z_sem_x1 = z_x1[:, :split]
        z_syn_x1 = z_x1[:, split:]

        z_sem_x2 = z_x2[:, :split]
        z_syn_x2 = z_x2[:, split:]

        # print('z_sem_x1.shape', z_sem_x1.shape)
        # print('z_sem_x2.shape', z_sem_x2.shape)
        # print('z_x1.shape', z_x1.shape)
        # print('z_x2.shape', z_x2.shape)

        z_x1_swapped = torch.cat((z_sem_x2, z_syn_x1), 1)
        z_x2_swapped = torch.cat((z_sem_x1, z_syn_x2), 1)

        # print('z_x1_swapped.shape', z_x1_swapped.shape)
        # print('z_x2_swapped.shape', z_x2_swapped.shape)
        # print()

        # input_x1 = torch.zeros(1, len(z_x1_swapped), dtype=torch.long, device=z_x1_swapped.device).fill_(self.vocab.go)
        # input_x2 = torch.zeros(1, len(z_x2_swapped), dtype=torch.long, device=z_x2_swapped.device).fill_(self.vocab.go)

        logits_x1, _ = self.decode(z_x1_swapped, x1)
        logits_x2, _ = self.decode(z_x2_swapped, x2)

        # print('logits_x1', logits_x1.shape)
        # print('y1', y1.shape)
        # print('logits_x1_orig', logits_x1_orig.shape)

        p_rec = self.loss_rec(logits_x1_orig, y1).mean()
        p_rec += self.loss_rec(logits_x2_orig, y2).mean()

        kl = loss_kl(mu_x1, logvar_x1) + loss_kl(mu_x2, logvar_x2)
        return {'p_rec' : p_rec, 'kl' : kl}
        # return {'rec': self.loss_rec(logits, targets).mean(),
        #         'kl': loss_kl(mu, logvar)}

    def loss(self, losses):
        return losses['p_rec'] + self.args.lambda_kl * losses['kl']

class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), nn.Sigmoid())
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
            self.args.lambda_p * losses['|lvar|']

    def autoenc(self, inputs, targets, is_train=False):
        _, logvar, z, logits = self(inputs, is_train)
        loss_d, adv = self.loss_adv(z)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, losses):
        super().step(losses)

        self.optD.zero_grad()
        losses['loss_d'].backward()
        self.optD.step()
