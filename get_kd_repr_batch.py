import torch
import esm
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter(truncation_seq_length=512)

device = torch.device('cuda:2')
model.to(device)
model.eval()  # disables dropout for deterministic results

# Load Data
df = pd.read_csv("../MLFF-DTA/dataset/Kd/data.csv")
print(df.head(4))
proteins = [(f"{idx}", df.loc[idx, "Sequence"]) for idx in range(len(df))]

# Create a DataLoader for batched protein processing
dataloader = DataLoader(proteins, batch_size=4, shuffle=False, collate_fn=lambda x: x)

counter = 0
for data in tqdm(dataloader):
    # Prepare data
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33].detach().cpu()
    contact_representations = results["contacts"].detach().cpu()
    del results
    
    for b, tokens_len in enumerate(batch_lens):
        sequence_representations = token_representations[b, 1 : tokens_len - 1].mean(0)
        contact_maps = contact_representations[b, : tokens_len, : tokens_len]
        # contact_maps = F.pad(contact_maps, [0, 512 - contact_maps.shape[0], 0, 512 - contact_maps.shape[1]])
        # print(sequence_representations.shape, contact_maps.shape)

        savedir = f"../MLFF-DTA/dataset_array/Kd/pretain-protein"
        os.makedirs(savedir, exist_ok=True)
        np.savez_compressed(
            f"{savedir}/sample_{counter}.npz",
            seq_repr = sequence_representations,
            concat_map = contact_maps,
        )
        # np.savez_compressed(
        #     f"{savedir}/sample_{counter}_concat.npz",
        #     concat_map = contact_maps,
        # )

        # if counter % 1000 == 0:
        #     plt.matshow(contact_maps.numpy())
        #     plt.savefig(f"att_concat_{counter}.png")
        #     plt.close()
        
        counter += 1


