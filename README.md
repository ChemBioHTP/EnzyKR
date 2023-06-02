The EnzyKR is the deep learning framework for the activation free energy prediction of the enzyme-substrate complexes.

## Getting Started
### Prerequisites
```bash
conda create -n kr python=3.8
conda install pytorch::pytorch 
conda install pandas numpy
pip install torch_geometric rdkit-pypi bidirectional_cross_attention
```

The model need to take the enzyme multiple sequence alignment, enzyme-substrate structural complexes as inputs. User can put the pdb files under the structures folder  and put the MSA files in a3m format under the msa folder. A3M files can be obtained from [HH-bilts](https://toolkit.tuebingen.mpg.de/tools/hhblits) webserver

Also a input of the SMILES string substrates need to provide under the raw folder  as the csv format. The example is shown under raw folder. However, the enzyme sequence column and dg++ column are not necessary. 

The model params need to download first from the google drive. The download instruction is under the model folder

And then run python scripts
```python
python preprocess.py
```

The output of the features are under the processed folder.

And running the inference model to predict the activation free energy.

```python
python inference.py --dataset ./ --model_path ./model/model1.pt
```


