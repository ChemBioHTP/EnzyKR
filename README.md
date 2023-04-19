The EnzyKR is the deep learning framework for the activation free energy prediction of the enzyme-substrate complexes .


if you want to use the pretrained model to predict the data please prepare the file as the shown in the input 
```python
$python inference.py ./test/data.tsv
```

if you want to retrain the model from the scratch. Please run the code with the following lines
```python
$python enzykr_bn.py 
```

## ToDo
- [x] Finshed the model.py cleaning with three different module graphNN, convolution, and attention part.
- [x] Finished the embedding module for both enzyme sequence and SMILES strings.
- [x] Finished the baseline model adopted from the DLKcat.

- [ ] Need to reformate the distance scripts for the enzyme-substrate interaction map generation. 
- [ ] Need to reformate the tool scripts for print the attention map in the model
- [ ] Need to reformate the dataloader in the model scripts
- [ ] Need to update the comparsion model for the binding SOTA affinity prediction
- [ ] Need to provide a inference script for the baseline model which can run the test set with params

Estimation finished by Friday with the following scripts




