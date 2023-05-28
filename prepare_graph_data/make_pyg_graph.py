import os
import torch
import pandas as pd
import pickle
import json

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from vocab_dict_class import VocabDict
from common import remove_string_literal

class pyg_graph_dataset(object):
    def __init__(self, name = ""):
        self.name = name
        self.vocab_dict = VocabDict()
        self.pyg_graphs = []


    def parse(self, graphs : list):

        written_num  = 0
        skip_num = 0
      
        for graph in graphs:
            try:
                pyg_data = from_networkx(graph)

                pyg_data = self.vocab_dict.update_vocab(pyg_data)

                pyg_data = Data(x = pyg_data.label, 
                                edge_index= pyg_data.edge_index,
                                edge_attr = pyg_data.type,
                                )

                self.pyg_graphs.append(pyg_data)
                written_num +=1
            except:
                skip_num+=1
                
        return written_num, skip_num
        
        
    def serialize(self, filename, dest = "./"):
        filename = filename.split(".")[0]
        # Graph data
        with open(os.path.join(dest, "graphs", filename + ".pickle"), "wb") as f:
            pickle.dump(self.pyg_graphs, f)
        
        # vocab dictionary
        with open(os.path.join(dest, "vocab_dict.pickle"), "wb") as f:
            pickle.dump(self.vocab_dict, f)