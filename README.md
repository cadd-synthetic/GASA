## GASA <br/>
### Description
* GASA (Graph Attention-based assessment of Synthetic Accessibility) is used to evaluate the synthetic accessibility of small molecules by distinguishing compounds to be easy- (ES) or hard-to-synthesize (HS).<br/>
* GASA focus on sampling around the decision boundary line and trained on 800,000 compounds from ChEMBL, GDBChEMBL and ZINC15 databases.<br/>
* GASA is graph neural network framework that makes self-feature deduction by applying an attention mechanism to automatically capture the most important structural features related to synthetic accessibility during the training process.<br/>
* GASA is able to identify structurally similar compounds effectively.<br/>
## Installation
### Known Installation Issues
#### The following versions must be used in order to use the pretrained models:
* python 3.6+ <br/>
* DGL 0.7.0+ [https://www.dgl.ai/pages/start.html]<br/>
* PyTorch 1.5.0+[https://pytorch.org/get-started/locally/]<br/>
* dgllife 0.2.6+ [https://github.com/awslabs/dgl-lifesci]<br/>
* RDKit (recommended version 2018.03.1+) [https://github.com/rdkit/rdkit]
### Use in Python
`
from gasa import GASA
smiles = 'NC(=O)OC[C@H](N)CC1=CC=CC=C1'
predict, pos, neg = GASA(smiles)
print(predict, pos, neg)
[0] [0.8078028559684753] [0.19219708442687988]
<br/><br/>

df = pd.read_csv('./test.csv')
smiles = list(df['smiles'])
predict, pos, neg = GASA(smiles)

`
 `gasa/main.py` for prediction. <br/>
Datasets used in GASA can be found in `gasa/data/data.zip` folder: dataset for training, validation and test the model. Three external test sets:TS1, TS2 and TS3
