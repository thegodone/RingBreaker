from rdkit import Chem
import pandas as pd
import sys
sys.path.insert(1, '../utils/')
from utils.ringbreakerbatch import Model

# Initialize the model
model = Model(dataset="uspto_ringbreaker", mask=False)


df = pd.read_csv('./data/reaxys_ringbreaker_extended.csv')
test_smiles= df['target'].tolist()
print(len(test_smiles))

# Get batch predictions
_, dfx = model.predict_ring_outcomes(test_smiles, cutoff=500)

print(dfx)

dfx.to_csv('results.csv',index=False)