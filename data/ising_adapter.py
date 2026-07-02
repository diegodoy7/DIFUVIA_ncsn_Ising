import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dotmap import DotMap

class IsingScoreDataset(Dataset):
    def __init__(self, data_dir, file_pattern='*.json'):
        self.data_pairs = [] 
        
        search_path = os.path.join(data_dir, file_pattern)
        file_list = glob.glob(search_path)
        
        if not file_list:
            raise RuntimeError(f"JSON files not found in {data_dir}")

        print(f"Procesando {len(file_list)} archivos de datos...")

        for file_path in file_list:
            # Logica de extraccion de temperatura
            filename = os.path.basename(file_path)
            try:
                temp_str = filename.replace('.json', '').split('_')[-1]
                temperature = float(temp_str)
            except ValueError:
                temperature = 0.0 

            with open(file_path, 'r') as f:
                content = json.load(f)
                for entry in content:
                    if 'config' in entry:
                        # MODIFICACION UNICA: Guardamos la tupla (config, temperatura)
                        self.data_pairs.append((entry['config'], temperature))

        print(f"Total de muestras cargadas: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # MODIFICACION UNICA: Desempaquetamos la tupla
        config_list, temp_val = self.data_pairs[idx]

        # 1. Convertir a Tensor
        x = torch.tensor(config_list, dtype=torch.float32)
        
        # 2. Anadir dimension de canal (1, 64, 64)
        x = x.unsqueeze(0) 

        # --- DEQUANTIZATION (Tu logica original intacta) ---
        
        # A) Mapear {-1, 1} -> {0, 1}
        x_norm = (x + 1) / 2.0
        
        # B) Anadir ruido uniforme u ~ U[0, 1]
        noise = torch.rand_like(x_norm)
        x_continuous = x_norm + noise 
        
        # C) Re-escalar a [-1, 1] aprox
        x_final = x_continuous - 1.0
        
        # MODIFICACION UNICA: Convertir temperatura a tensor
        temp_tensor = torch.tensor(temp_val, dtype=torch.float32)
        
        # D) Retornar (IMAGEN, TEMPERATURA)
        return x_final, temp_tensor

# --- FUNCION DE ENLACE PARA EL REPOSITORIO ---
def get_dataloader(config):
    dataset = IsingScoreDataset(data_dir=config.data.path)
    # Se recomienda num_workers > 0 si estas en Linux/Mac para velocidad
    num_workers = getattr(config.data, 'num_workers', 0)
    
    return DataLoader(
        dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
