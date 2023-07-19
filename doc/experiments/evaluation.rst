Evaluation
##########

For the purpose of evaluating the methods on the real-world data sets and comparing the results with the papers proposing the methods, we adopted their metrics to build our prediction scores.
First, we used two different measures of predictions i.e. accuracy and sensitivity x specificity.
Sensitivity is the proportion of correctly classified positives while latter one denotes the proportion of correctly classified negatives (so the recall of positives and the recall of negatives).
The predictions of the lazy learning methods are obtained by a Naive Bayes, which is already implemented in the methods, while the predictions of [todo: smilla].
Second, we measure the performance of feature selection by the compression of the feature number, so the ratio of selected features to all features.

The hyperparameter k of lazy learning, that is the maximum number of selected features, was chosen in accordance to the papers with k = [30, 40, 50].
The gene data set was divided in a train-test set on a ratio of 70/30, since the data set is relatively small and the lazy learning approach is executed per testing instance, which should not be too less.