import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import json
import numpy as np
import re
from argparse import Namespace

class IsingScoreDataset(Dataset):
    def __init__(self, root_dir, channels=1):
        self.files = glob.glob(os.path.join(root_dir, '**/*.json'), recursive=True)
        self.data_list = []
        self.temp_list = []
        
        print(f" Scanning {len(self.files)} files...")

        for file_path in self.files:
            try:
                filename = os.path.basename(file_path)
                temp_str = filename.replace('.json', '').split('_')[-1]
                temp_val = float(temp_str)

                
                with open(file_path, 'r') as f:
                    content = json.load(f) 

                lattices = [item['config'] for item in content]
                batch_imgs = torch.tensor(lattices, dtype=torch.float32).unsqueeze(1) 
                
                # temperature tensor
                batch_temps = torch.full((len(lattices),), temp_val, dtype=torch.float32)

                self.data_list.append(batch_imgs)
                self.temp_list.append(batch_temps)
                
            except Exception as e:
                print(f" Error en {filename}: {e}")

        # merge all batches into single tensors
        if self.data_list:
            self.data = torch.cat(self.data_list, dim=0)
            self.temps = torch.cat(self.temp_list, dim=0)
            print(f"{len(self.data)} samples loaded.")
        else:
            raise ValueError("Data loading failed: No valid files found")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.temps[idx]

def get_dataloader(config):
    dataset = IsingScoreDataset(root_dir=config.data.path)
    return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=False)


def dict2namespace(d):
    namespace = Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace