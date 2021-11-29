# DrugInteractions

### Directory Structure
```bash
├── data
│   ├── ddi_pairs.txt
├── dataloaders.py
├── main.py
├── models.py
├── utils.py
└── .gitignore
```

### Requirements
- Python3
- pandas
- numpy
- torch
- tensorboardX

### Running the training script
If the DDI pairs data lives in `./data/ddi_pairs.txt`, then you might run something like this:
```
python main.py --data_fn './data/ddi_pairs.txt' --hid_dim 256 --epochs 300 --savedir './results/'  --exp_name 'baseline_256h'  --test_epoch 1  --batch_size 1024  --cuda --save
```

To run the training script with graph structure, you might run something like this:
```
python main_graph.py --cuda --batch_size 1024 --test_epoch 1 --hid_dim 512 --model GCNEntPair --save  --exp_name 'gcnentpair_512h'
```

Modify the `savedir` and `exp_name` to wherever you'd like your results to be stored. If the directory specified by `savedir` does not currently exist, [`utils.setup_experiment_log`](https://github.com/horacepan/DrugInteractions/blob/d9d292737815f827d1d1a7b2136363f80da4866e/utils.py#L38) will create it.
