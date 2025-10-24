import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter(truncation_seq_length=1024)
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MRFATSTIVKVALLLSSLCVDAAVMWNRDTSSTDLEARASSGYRSVVYFVNWAIYGRNHNPQDLPVERLTHVLYAFANVRPETGEVYMTDSWADIEKHYPGDSWSDTGNNVYGCIKQLYLLKKQNRNLKVLLSIGGWTYSPNFAPAASTDAGRKNFAKTAVKLLQDLGFDGLDIDWEYPENDQQANDFVLLLKEVRTALDSYSAANAGGQHFLLTVASPAGPDKIKVLHLKDMDQQLDFWNLMAYDYAGSFSSLSGHQANVYNDTSNPLSTPFNTQTALDLYRAGGVPANKIVLGMPLYGRSFANTDGPGKPYNGVGQGSWENGVWDYKALPQAGATEHVLPDIMASYSYDATNKFLISYDNPQVANLKSGYIKSLGLGGAMWWDSSSDKTGSDSLITTVVNALGGTGVFEQSQNELDYPVSQYDNLRNGMQT"),
    ("protein2", "LLLSSLCVDAAVMWNRDT")
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]
print(token_representations.shape)
print(results.keys())

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
print(sequence_representations[0].shape)

# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (seq_name, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title(seq[0:10] + " ...")
    # plt.show()
    plt.savefig(f"attn_contact_{seq_name}.png")
    plt.close()

# concat_map = attention_contacts[: tokens_len, : tokens_len].numpy()
# print(concat_map.min(), concat_map.max())

# # 转换为图
# graph_data = contact_map_to_graph(contact_map, sequence)

# print("节点特征矩阵形状:", graph_data.x.shape)
# print("边索引矩阵形状:", graph_data.edge_index.shape)
# print("边属性:", graph_data.edge_attr)