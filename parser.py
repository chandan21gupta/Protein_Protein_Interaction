import numpy as np
from Bio import SeqIO
import pandas as pd
record_dict = SeqIO.to_dict(SeqIO.parse("./dataset/viral_protein_1.fasta", "fasta"))
protein_dict = pd.read_csv("ProtVec.csv", sep = '\t')

dataset = []

for keys in record_dict:
	sequence_vector = np.zeros(shape = (1,100))
	seq = record_dict[keys].seq
	for i in range(len(seq)-2):
		word = seq[i:i+3]
		vector = protein_dict[protein_dict['words'] == word].to_numpy()
		sequence_vector = sequence_vector+np.delete(vector,0,axis = 1)
	sequence_vector = np.squeeze(sequence_vector, axis = 0)
	dataset.append(sequence_vector)
final_dataset = np.array(dataset)
print(final_dataset.shape)