# eight-ml-classifiers



**Eight Model Performance:**

For more information like model parameters please see `workflow-part1.ipynb` and `workflow-part2.ipynb`.



|          | Training Log  Loss | Testing Log Loss | Training  Accuracy | Testing  Accuracy | Training F1  Score | Testing F1  Score |
| -------- | ------------------ | ---------------- | ------------------ | ----------------- | ------------------ | ----------------- |
| XGBoost  | 0.451              | 1.000            | 0.852              | 0.507             | 0.892              | 0.651             |
| SVM      | 0.963              | 1.272            | 1.000              | 0.505             | 1.000              | 0.603             |
| Voting   | 0.338              | 1.013            | 0.992              | 0.487             | 0.998              | 0.659             |
| kNN      | 0.000              | 4.729            | 1.000              | 0.484             | 1.000              | 0.617             |
| AdaBoost | 2.181              | 2.184            | 0.976              | 0.479             | 0.970              | 0.648             |
| RF       | 0.048              | 1.343            | 0.992              | 0.479             | 0.996              | 0.635             |
| MLP      | 0.591              | 1.095            | 0.632              | 0.450             | 0.825              | 0.612             |
| LR       | 0.893              | 1.178            | 0.570              | 0.426             | 0.704              | 0.574             |