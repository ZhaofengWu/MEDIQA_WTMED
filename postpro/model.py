import torch
import torch.nn as nn
import torch.nn.functional as F

class PostProModel(nn.Module):
    """
    Postprocessing model
    """
    def __init__(self, n_classes, n_members, hidden_dim):
        super(PostProModel, self).__init__()
        self.n_classes = n_classes
        self.n_members = n_members

        input_dim = n_classes + 1  # for position encoding
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, prob):
        assert len(prob.shape) == 3 and prob.shape[1] == self.n_members and prob.shape[2] == self.n_classes
        new_prob = torch.FloatTensor(prob.shape[0], self.n_members, self.n_classes + 1).to(prob.device)
        new_prob[:,:,:self.n_classes] = prob
        new_prob[:,:,self.n_classes] = torch.arange(self.n_members)
        prob = new_prob

        hidden = F.relu(self.linear1(prob))  # (bsz, n_members, hidden_dim)

        attn = hidden.bmm(hidden.transpose(1, 2))  # (bsz, n_members, n_members)
        attn = F.softmax(attn, dim=-1)
        attended_hidden = attn.bmm(hidden)  # (bsz, n_members, hidden_dim)

        return self.linear2(attended_hidden)
