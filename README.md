# **SECRET**

<img width="837" height="464" alt="Screenshot 2025-08-11 at 10 11 26 AM" src="https://github.com/user-attachments/assets/32eedfb1-2f49-4de2-8236-214496d654f6" />


# Data
Find the demo train data in ```data``` folder. We have also shared a full list of unlabeled and labeled trials we used for training and validation in train_trials_unlabeled.csv, train_trials_labeled.csv and val_trials.csv respectively . Please download data from ```https://aact.ctti-clinicaltrials.org```. Test data is already in the folder named as ```test_data.csv```.


# *Run SECRET on Test Data*: 

```python3 inference.py```


# Trainining process

1. *To use LLM to generate Q/A pairs go to*:
   
```cd q_a```

```python3 q_a_generation_arg.py -n criteria -f demo_train_data.csv```

2. *Cleaning LLM generated data*:

run ```clean_data.ipynb```,```predefined_question.ipynb```

3. *Create positive and negative*:

```python3 create_positive_pairs_q_a_level.py```

```python3 create_positive_pairs_trial_level.py``` 

4. *To do local contrastive training*: ```python3 local.py  -p 'data/text_q_a_level.json' -v 'data/val_data.csv' -l 'data/val_list.csv' -a True -m 'recall@5'```

5. *To do global contrastive training*: ```python3 global.py -f data/text_trial_level.json -v data/val_data.csv -l data/val_list.csv -a True -m recall@5```

*Our best model is available at* 
```models/global_model.pth```


## BibTeX Citation

```bibtex
@inproceedings{das-etal-2025-secret,
    title = "$SECRET$: Semi-supervised Clinical Trial Document Similarity Search",
    author = "Das, Trisha  and
      Shafquat, Afrah  and
      Beigi, Mandis  and
      Aptekar, Jacob  and
      Sun, Jimeng",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.264/",
    doi = "10.18653/v1/2025.acl-long.264",
    pages = "5278--5291",
    ISBN = "979-8-89176-251-0"
}
```


