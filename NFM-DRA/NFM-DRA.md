# NFM-DRA (Normal feature memory-enhanced DRA)

## Getting started (taking the fundus dataset as an example)

1. Prepare the environment following [FUNAD](https://github.com/SpatialAILab/FUNAD?tab=readme-ov-file#prerequisites)

2. Train DRA by
```
bash NFM-DRA/DRA/train_fundus.sh
```

3. Evaluate the performance of DRA on the test set
```
bash NFM-DRA/DRA/eval_fundus.sh
```

4. Train NFM by
```
bash NFM-DRA/NFM/fundus_training.sh
```

5. Evaluate the performance of NFM-DRA
```
bash NFM-DRA/NFM-DRA_evaluate.sh
```