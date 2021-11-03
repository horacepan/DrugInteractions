# DrugInteractions

### Directory Structure
```bash
├── data
│   ├── ddi
│   │   ├── ddi_pairs.txt
├── results
│   ├── ddi (directory to save experiment logs/summary writer outputs/checkpoints)
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
If the DDI pairs data lives in './data/ddi_pairs.txt', then you run something like this:
```
python main.py --data_fn './data/ddi_pairs.txt' --hid_dim 256 --epochs 300 --savedir './results/ddi/'  --exp_name 'baseline_256h'  --test_epoch 1  --batch_size 1024  --cuda --save
```

Modify the `savedir` and `exp_name` to wherever you'd like your results to be stored.
