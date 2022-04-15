## GASA <br/>
### Description
* GASA (Graph Attention-based assessment of Synthetic Accessibility) is a Python package to evaluate the synthetic accessibility of small molecules by distinguishing compounds to be easy- (ES) or hard-to-synthesize (HS).<br/>
* GASA focus on sampling around the decision boundary line and trained on 800,000 compounds from ChEMBL, GDBChEMBL and ZINC15 databases. (Data can be found in data.zip)<br/>
* GASA is graph neural network framework that makes self-feature deduction by applying an attention mechanism to automatically capture the most important structural features related to synthetic accessibility during the training process.<br/>
* GASA is able to identify structurally similar compounds effectively.<br/>
## Installation
### Known Installation Issues
#### Usage
``` 
from rdkit import Chem
from syba.syba import SybaClassifier

syba = SybaClassifier()
syba.fitDefaultScore()
smi = "O=C(C)Oc1ccccc1C(=O)O"
syba.predict(smi)
# syba works also with RDKit RDMol objects
mol = Chem.MolFromSmiles(smi)
syba.predict(mol=mol)
# syba.predict is actually method with two keyword parameters "smi" and "mol", if both provided score is calculated for compound defined in "smi" parameter has the priority
syba.predict(smi=smi, mol=mol) 
```
#### The following versions must be used in order to use the pretrained models:
python 3.6+ <br/>
DGL 0.7.0+ [https://www.dgl.ai/pages/start.html]<br/>
PyTorch 1.5.0+[https://pytorch.org/get-started/locally/]<br/>
dgllife 0.2.6+ [https://github.com/awslabs/dgl-lifesci]<br/>
RDKit (recommended version 2018.03.1+) [https://github.com/rdkit/rdkit]
