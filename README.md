# KNN from Scratch — Wine Classification

A **K-Nearest Neighbors classifier implemented with NumPy**, covering Euclidean distance, neighbor selection, class voting, feature standardization, and accuracy evaluation. The project demonstrates the algorithm on the Wine dataset and achieves **94.44% test accuracy** with a neighbor count selected through training-set cross-validation.

## Highlights

- **Transparent implementation:** Follow the algorithm from distance calculations to the final predicted label.
- **Vectorized distances:** Calculate distances to every training sample in one NumPy operation per query.
- **Reusable functions:** Predict one sample or a batch using explicit data arguments.
- **Training-only preprocessing:** Learn standardization statistics separately inside each cross-validation fold.
- **Deterministic voting:** Resolve tied class votes using nearest representatives, with a stable rule for equal distances.
- **Input validation:** Check feature dimensions, finite values, label lengths, and valid neighbor counts.
- **Clear evaluation:** Compare validation scores, inspect per-class precision/recall/F1, and visualize a confusion matrix.

## Results

The experiment uses a **stratified 70/30 split** with `random_state=42`: **124 training samples** and **54 test samples**.

### Selecting k

Five-fold stratified cross-validation compares the original candidates `3`, `5`, and `7` within the training set. Each fold fits its own standardization statistics using only its training rows.

| k | Mean validation accuracy | Standard deviation across folds |
| ---: | ---: | ---: |
| 3 | 95.97% | 4.00 percentage points |
| **5** | **96.80%** | **3.35 percentage points** |
| 7 | 96.80% | 3.35 percentage points |

Both `k=5` and `k=7` have the same mean validation accuracy. The selection rule prefers the smaller `k`, so **k=5** is chosen. Test scores are not used to select `k` in this workflow.

### Test evaluation

| Metric | Result |
| --- | ---: |
| Selected neighbor count | 5 |
| Test accuracy | **94.44%** |
| Correct predictions | **51 / 54** |

Confusion matrix, with actual classes as rows and predicted classes as columns:

| Actual / Predicted | Class 0 | Class 1 | Class 2 |
| --- | ---: | ---: | ---: |
| Class 0 | 18 | 0 | 0 |
| Class 1 | 0 | 18 | 3 |
| Class 2 | 0 | 0 | 15 |

Results describe this fixed Wine dataset split. The notebook includes the fold scores, full classification report, and evaluation plots.

## How the classifier works

1. **Standardize inputs:** Subtract each feature’s training mean and divide by its training standard deviation.
2. **Calculate distances:** Measure Euclidean distance between the query and every standardized training row.
3. **Select neighbors:** Sort distances and keep the closest `k` examples.
4. **Count votes:** Count the class labels among those neighbors.
5. **Resolve ties:** Among classes tied for the most votes, select the one whose nearest representative is closest. Equal distances preserve training-row order.
6. **Return the prediction:** Produce the winning class label.

KNN stores training examples rather than learning model weights. This implementation uses uniform voting: every selected neighbor contributes one vote.

### Feature scaling

Wine measurements have different numerical scales. Standardization prevents a feature from dominating distance calculations simply because its values are numerically larger.

The NumPy implementation uses the population standard deviation (`ddof=0`) and a scale of 1 for constant training features. Validation and test rows always use statistics learned from their corresponding training partition.

### What is implemented from scratch?

| Component | Implementation |
| --- | --- |
| Euclidean distance | NumPy |
| Neighbor search and selection | NumPy distance calculations and stable sorting |
| Class voting and tie-breaking | Python and `collections.Counter` |
| Standardization | NumPy means and standard deviations |
| Accuracy calculation | NumPy label comparison |
| Dataset loading and split/fold indices | scikit-learn |
| Classification report and confusion matrix | scikit-learn |
| Plots | Matplotlib |

The notebook does not use scikit-learn’s `KNeighborsClassifier` or `StandardScaler`.

## Dataset

The [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) contains **178 samples**, **13 numerical features**, and **three classes**, based on chemical measurements of wines from three cultivars. Features include alcohol, malic acid, flavanoids, color intensity, and proline.

The notebook loads the dataset through [scikit-learn’s `load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). It is bundled with scikit-learn, so no separate dataset download or local data file is required.

This project predicts the wine class, not a wine-quality score.

## Getting started

Download the project files into one folder, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook knn.ipynb
```

Run all cells from top to bottom. The notebook loads the data, defines the classifier and standardizer, selects `k` through cross-validation, and evaluates the selected configuration. No GPU is needed.

## Using the functions

After running the function-definition cells, fit scaling on your training data and reuse it for new samples:

```python
mean, scale = fit_standardizer(X_train)
train_scaled = standardize(X_train, mean, scale)
test_scaled = standardize(X_test, mean, scale)

predicted_labels = get_predictions(best_k, train_scaled, test_scaled, y_train)
accuracy = evaluate(predicted_labels, y_test)

single_prediction = knn(best_k, train_scaled, test_scaled[0], y_train)
```

Use the same feature order in training and prediction inputs. `best_k` is set by the notebook’s cross-validation cell.

## References

- [UCI Machine Learning Repository — Wine](https://archive.ics.uci.edu/dataset/109/wine).
- [scikit-learn — Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- [NumPy documentation](https://numpy.org/doc/stable/).
- [KNN from Scratch — reference implementation credited in the notebook](https://github.com/mavaladezt/kNN-from-Scratch/blob/master/knn.py).

