import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://github.com/ttpro1995/TreeLSTMSentiment/blob/master/model.py
# original version in Lua: https://github.com/stanfordnlp/treelstm/blob/master/models/BinaryTreeLSTM.lua
# this file implements the gate_output == False version, since we don't have intermediate supervision

class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """
    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)

    def forward(self, input):
        c = self.cx(input)
        h = F.tanh(c)
        return c, h

class BinaryTreeComposer(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})
    """
    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        self.olh, self.orh = new_gate()

    def forward(self, lc, lh , rc, rh, output_gate=False):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = F.tanh(self.ulh(lh) + self.urh(rh))
        c = i * update + lf * lc + rf * rc
        o = F.sigmoid(self.olh(lh) + self.orh(rh)) if output_gate else 1
        h = o * F.tanh(c)
        return c, h, i

class BinaryTreeLSTM(nn.Module):
    def __init__(self, mem_dim, embedding_matrix, token2idx, unked_words=None):
        super(BinaryTreeLSTM, self).__init__()
        self.in_dim = embedding_matrix.shape[1]
        self.mem_dim = mem_dim
        self.token2idx = token2idx
        self.unked_words = unked_words if unked_words is not None else set()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], self.in_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)

        self.leaf_module = BinaryTreeLeafModule(self.in_dim, mem_dim)
        self.composer = BinaryTreeComposer(self.in_dim, mem_dim)

    def forward(self, tree, outmost_call=True):
        """
        tree: Tree object (my version)
        """
        if len(tree.children) == 0:
            # leaf case
            idx = self.token2idx[tree.content.lower() if tree.content.lower() not in self.unked_words else '<unk>']
            idx = torch.LongTensor([idx])[0].to(self.embedding.weight.device)
            return self.leaf_module(self.embedding(idx)) + (None,)
        elif len(tree.children) == 1 and outmost_call and len(tree.children[0].children) == 0:  # allow single child only for 1-ele trees
            return self(tree.children[0], outmost_call=False)
        else:
            assert len(tree.children) == 2
            lc, lh, _ = self(tree.children[0], outmost_call=False)
            rc, rh, _ = self(tree.children[1], outmost_call=False)
            return self.composer(lc, lh, rc, rh, output_gate=outmost_call)
