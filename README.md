# **SECRET**

*To run SECRET on Test Data*:

```python3 inference.py```

*To use LLM to generate Q/A pairs go to*:

```cd q_a```

```python3 q_a_generation_arg.py -n criteria -f train_data.csv```

*Preprocessing data*:

run ```clean_data.ipynb``` and ```predefined_question.ipynb```

*To do local contrastive training*:

```python3 local.py  -p 'data/text_q_a_level.json' -v 'data/val_data.csv' -l 'data/val_list.csv' -a True -m 'recall@5'```

*To do global contrastive training*: 

```python3 global.py -f data/text_trial_level.json -v data/val_data.csv -l data/val_list.csv -a True -m recall@5```

*Our best model is available at* 

```models/global_model.pth```




