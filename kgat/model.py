import torch
from torch import nn

class TransR(nn.Module):
    """
    TransR model : Entity and Relation Embeddings for Knowledge Graph Completion
    """
    def __init__(self, config):
        super(TransR, self).__init__()
        self.config = config
        self.emb_e = nn.Embedding(config.nentity, config.edim)
        self.emb_rel = nn.Embedding(config.nrelation, config.rdim)
        self.ent_transfer = nn.Linear(config.edim, config.rdim, bias=False)
        self.emb_rel.weight = nn.Parameter(torch.diag(torch.ones(config.nrelation)))
        self.init()
    
    def init(self):
        nn.init.xavier_uniform_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel.weight.data)

    def forward(self, h, rel, t):
        e_h = self.emb_e(h)
        e_r = self.emb_rel(rel)
        e_t = self.emb_e(t)
        plausibility_score = torch.norm(self.ent_transfer(e_h) + e_r - self.ent_transfer(e_t), self.config.p_norm, -1)
        # return plausibility**p_norm
        return torch.pow(plausibility_score, self.config.p_norm)
    
    def get_ent_embeddings(self):
        return self.emb_e.weight.data.cpu().numpy()
    
    def get_rel_embeddings(self):
        return self.emb_rel.weight.data.cpu().numpy()
