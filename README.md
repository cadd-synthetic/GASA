## GASA: Synthetic Accessibility Prediction of Organic Compounds based on Graph Attention Mechanism <br/>
### Description
* GASA (Graph Attention-based assessment of Synthetic Accessibility) is used to evaluate the synthetic accessibility of small molecules by distinguishing compounds to be easy- (ES, 0) or hard-to-synthesize (HS, 1).<br/>
* GASA focus on sampling around the hypothetical decision boundary line and trained on 800,000 compounds from ChEMBL, GDBChEMBL and ZINC15 databases.<br/>
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

## Use in Python
```
from gasa import GASA 
smiles = 'NC(=O)OC[C@H](N)CC1=CC=CC=C1' 
predict, pos, neg = GASA(smiles) 
print(predict, pos, neg) 
[0] [0.8078028559684753] [0.19219708442687988] 


df = pd.read_csv('./test.csv')
smiles = list(df['smiles'])
print(smiles)
[CCOC(=O)c1c(NC(C)=O)sc2c1CCN(Cc1ccccc1)C2,
CCOC(=O)c1ccc2[nH]c(-c3ccc(C)cc3)nc2c1,
CCc1[nH]cc2c1c(CO)cc1nc[nH]c12,
CC1=C(C)C2(CC2)C(C(C)(C)C(C)(C)[NH3+])C1]

predict, pos, neg = GASA(smiles)
print(predict, pos, neg) 
[0, 0, 1, 1]
[0.9403825402259827, 0.8335544466972351, 0.19376544654369354, 0.1610676646232605]
[0.05961743742227554, 0.1664455384016037, 0.8062344789505005, 0.8389323353767395]
```
## The structure of the code is as follows:
In data: <br/>
 * Dataset for training, validation and test the model <br/>
 * three external test sets:TS1, TS2 and TS3 <br/>
 
In model: <br/> 
 * data.py: import and process the data <br/>
 * model.py: define GASA models <br/>
 * gasa_utils.py: converts SMILES into graph with features <br/>
 * hyper.py: code for hyper-parameters optimization <br/>
 * gasa.pth: saved pretrained model <br/>
 * gasa.json: best combination of hyper-parameters for GASA <br/>

Outside: <br/>
* gasa.py: code for predicting the results for given molecules <br/>
* test.csv: several molecules for test the model
* explain.ipynb: atom weights visualization for given compound <br/>

## Citation
Jiahui Yu; Jike Wang; Hong Zhao; Junbo Gao; Yu Kang; Dongsheng Cao; Zhe Wang; Tingjun Hou. Synthetic Accessibility Prediction of Organic Compounds Based on the Graph Attention Mechanism. *J. Chem. Inf. Model.* 2022 <br/>
https://doi.org/10.1021/acs.jcim.2c00038
