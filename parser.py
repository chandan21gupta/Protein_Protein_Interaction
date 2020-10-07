import os
import numpy as np
from Bio import SeqIO
import pandas as pd
record_dict = SeqIO.to_dict(SeqIO.parse("./dataset/viral_protein_1.fasta", "fasta"))
protein_dict = pd.read_csv("ProtVec.csv", sep = '\t')

dataset = []

# for keys in record_dict:
# 	sequence_vector = np.zeros(shape = (1,100))
# 	seq = record_dict[keys].seq
# 	for i in range(len(seq)-2):
# 		word = seq[i:i+3]
# 		vector = protein_dict[protein_dict['words'] == word].to_numpy()
# 		sequence_vector = sequence_vector+np.delete(vector,0,axis = 1)
# 	sequence_vector = np.squeeze(sequence_vector, axis = 0)
# 	dataset.append(sequence_vector)
# final_dataset = np.array(dataset)
# print(final_dataset.shape)

def get_vector(sequence_1, sequence_2):
	sequence_vector_1 = np.zeros(shape = (1,100))
	for i in range(len(sequence_1)-2):
		word = sequence_1[i:i+3]
		vector = protein_dict[protein_dict['words'] == word].to_numpy()
		sequence_vector_1 = sequence_vector_1+np.delete(vector, 0, axis = 1)
	sequence_vector_1 = np.squeeze(sequence_vector_1, axis = 0)
	sequence_vector_2 = np.zeros(shape = (1,100))
	for i in range(len(sequence_2)-2):
		word = sequence_2[i:i+3]
		vector = protein_dict[protein_dict['words'] == word].to_numpy()
		sequence_vector_2 = sequence_vector_2+np.delete(vector, 0, axis = 1)
	sequence_vector_2 = np.squeeze(sequence_vector_2, axis = 0)
	final_sequence = np.add(sequence_vector_1, sequence_vector_2)
	return final_sequence


def final_dataset(sample = 'positive'):
	path = ''
	if sample == "positive":
		path = './dataset/'
	else:
		path = './Dataset_Negative/'

	files = os.listdir(path)
	dataset = []
	for file in files:
		record_dict = SeqIO.to_dict(SeqIO.parse(path+file, "fasta"))
		viral_protein = ''
		human_protein = []
		for keys in record_dict:
			name = record_dict[keys].name
			if sample == 'positive':
				if file == 'viral_protein_2.fasta' and (name == 'sp|P0DTC5|VME1_SARS2'):
					viral_protein = record_dict[keys].seq
				elif file != 'viral_protein_2.fasta' and ("HUMAN" not in name):
					viral_protein = record_dict[keys].seq
				else:
					human_protein.append(record_dict[keys].seq)
			else:
				if ("SARS" in name or "MERS" in name or "VIRUS" in name):
					viral_protein = record_dict[keys].seq
				else:
					human_protein.append(record_dict[keys].seq)
		for protein in human_protein:
			features = get_vector(viral_protein, protein)
			dataset.append(features)
	dataset = np.vstack(dataset)
	return dataset

# files = os.listdir('./dataset/')
# positive_viral_proteins = []
# for file in files:
# 	record_dict = SeqIO.to_dict(SeqIO.parse('./dataset/'+file, "fasta"))
# 	for keys in record_dict:
# 		name = record_dict[keys].name
# 		if file == "viral_protein_2.fasta" and (name == "sp|P0DTC5|VME1_SARS2"):
# 			positive_viral_proteins.append(name)
# 		elif file != "viral_protein_2.fasta" and ("HUMAN" not in name):
# 			positive_viral_proteins.append(name)

# files = os.listdir('./Dataset_Negative/')
# negative_viral_proteins = []
# for file in files:
# 	record_dict = SeqIO.to_dict(SeqIO.parse('./Dataset_Negative/'+file, "fasta"))
# 	for keys in record_dict:
# 		name = record_dict[keys].name
# 		if ("SARS" in name or "MERS" in name or "VIRUS" in name):
# 			negative_viral_proteins.append(file)
			
# print(len(negative_viral_proteins))

#generate datasets
positive_samples = final_dataset(sample = 'positive')
negative_samples = final_dataset(sample = 'negative')
print(positive_samples.shape)
print(negative_samples.shape)

np.save('positive_samples', positive_samples)
np.save('negative_samples', negative_samples)