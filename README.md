# GASA <br/>
GASA (Graph Attention-based assessment of Synthetic Accessibility) is a Python package to evaluate the synthetic accessibility of small molecules by distinguishing compounds to be easy- (ES) or hard-to-synthesize (HS).<br/>
GASA focus on sampling around the decision boundary line and trained on 800,000 compounds from ChEMBL, GDBChEMBL and ZINC15 databases.(Data can be found in data.zip)<br/>
GASA is graph neural network framework that makes self-feature deduction by applying an attention mechanism to automatically capture the most important structural features related to synthetic accessibility during the training process.<br/>
GASA is able to identify structurally similar compounds effectively.<br/>
# Installation
# Known Installation Issues
## The following versions must be used in order to use the pretrained models:
python 3.6+, DGL 0.7.0+, PyTorch 1.5.0+, dgllife 0.2.6+ and RDKit(recommended version 2018.03.1+).
