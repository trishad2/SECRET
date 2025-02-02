# **SECRET**


# Data
Find the demo train data in ```data``` folder. We have also shared a full list of unlabeled and labeled trials we used for training, testing and validation in train_trials.csv, test_trials.csv and val_trials.csv respectively . Please download data from ```https://aact.ctti-clinicaltrials.org```.


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




