from torch import nn
from torch.utils.data import Dataset


class KGATDataset(Dataset):
    """
    Defines the dataset class for Knowledge Graph Attention Networks
    """
    def __init__(self, ckg_onto):
        """
        :param ckg_onto: the ontology of the collaborative knowledge graph:
            A graph with the knowledge graph and the interaction between items and users
        """
        self.ckg = ckg_onto
        