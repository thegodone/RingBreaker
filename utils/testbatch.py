from rdkit import Chem

import sys
sys.path.insert(1, '../utils/')
from utils.ringbreakerbatch import Model

# Initialize the model
model = Model(dataset="uspto_ringbreaker", mask=False)

# Test with a small batch
test_smiles = ['CC1=CC=CC=C1', 'CCOC(=O)C1=CC=CC=C1','C1=CC2=CC=CN=C2C=C1']

# Get batch predictions
batch_results = model.predict_ring_outcomes(test_smiles, cutoff=10)

# Display results
for i, res in enumerate(batch_results):
    print(f"Results for SMILES: {test_smiles[i]}")
    print(res)
    print("\n")