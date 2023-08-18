###########
Experiments
###########

In order to evaluate the implemented methods and compare them with each other we conducted a number of experiments.
To allow further comparisons to the papers introducing the algorithms we focused on following their set-up.

Datasets
=========

For the purpose of evaluating the implemented methods we use two different datasets, following the mentioned papers about lazy and eager learning approaches. 
For lazy learning we use a bioinformatic dataset and for the eager approach the SportsTweets dataset. 

Bioinformatic dataset
**********************

For the purpose of evaluating the lazy methods we follow Wan, Freitas and de Magalhaes' task of predicting a *C. elegans* gene’s effect on the organism’s longevity. 
The corresponding information is obtained from the database of Human Ageing Genomic Resources :cite:p:`TacutuRobi2013HAGR`.
Hierarchical relations are pictured in the Gene Ontology (GO) :cite:p:`GeneOntologyConsortium2004` from the National Institutes of Health. 

The GO collects terms and hierarchical relationships in a "is-a" generalizations/specializations, among those, building a directed acyclic graph.
If a gene is associated with the GO term i, it is also associated with all ancestors of i in the hierarchy (assured in the preprocessing of instances).
The gene2go :cite:p:`gene2go` maps the GO-terms to the associated genes.
We build our dataset by selecting all EntrezID numbers of each *C. elegans* gene in the HAGR database with the same EntrezID number in the NCBI gene database gene2go [2023-06-25]. 
Then the GO terms for matched genes are mapped to the data from the gene2go database. Thus, we obtain a dataset with 889 genes (rows) and 27416 go-terms (columns) set to 1 if the go-term is present in this gene.
After deleting genes associated with no go-terms the dataset includes 819 genes and the final set with only genes which effect on longevity is known, the dataset contains 793 rows.

SportsTweets
************

The SportsTweets dataset [1]_ which we primarily use for evaluating the eager prediction methods is inspired by the *Sports Tweets T* dataset used
in the paper proposing the SHSEL Feature Selection algorithm by :cite:authors:`ristoski2014feature`. It consists of tweets which are classified into
sports related and non-sports related tweets. The dataset is linked to DBpedia using the 
DBpedia Spotlight service. For the feature selection and classification tasks we use the extracted DBpedia entitites and types which we generate
for the entitites using the `direct_type_generator <https://kgextension.readthedocs.io/en/latest/source/usage_generators.html#direct-type-generator>`_ 
from the `kgextension library <https://github.com/om-hb/kgextension>`_. This method also generates the corresponding hierachy which is 
build using hyponymy and hypernymy relationships between the types. For computational reasons we only use a subset including the first 300 samples 
of the dataset. 


Experiments Setup and Evaluation
=================================

Eager Learning
***************
For the eager learning approach we evaluate on both datasets. We split the dataset into 70% for training and 30% for evaluation. 
We then fit the feature selectors on the training set and transform both training and eval set using the features selected on 
the training set. For all feature selectors the default hyperparameters are used. These default values are either based on the 
papers in which the methods are proposed or chosen heuristically if the papers don't mention which values should be used.
After feature selection, we fit a `Naive Bayes classifier <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html>`_ 
implemented in scikit-learn. Here we also keep the default hyperparameters. We finally evaluate the classifier on the eval set.
We evaluated the effects of the feature selection by computing the classification accuracy and the feature compression rate. The 
scores are compared with a baseline where we classify the dataset without performing feature selection.


Lazy Learning
**************

For the purpose of evaluating the methods on the real-world datasets and comparing the results with the papers proposing the methods, we adopted their metrics to build our prediction scores.
First, we used two different measures of predictions i.e. accuracy and sensitivity x specificity.
Sensitivity is the proportion of correctly classified positives while latter one denotes the proportion of correctly classified negatives (so the recall of positives and the recall of negatives).
The predictions of the lazy learning methods are obtained by a Naive Bayes, which is already implemented in the methods.
Second, we measure the performance of feature selection by the compression of the feature number, so the ratio of selected features to all features.

The hyperparameter k of lazy learning, that is the maximum number of selected features, was chosen in accordance to the papers with k = [30, 40, 50].

The gene data set was divided in a train-test set on a ratio of 70/30, since the data set is relatively small and the lazy learning approach is executed per testing instance, which should not be too less.

Results
========
.. csv-table:: Lazy learning
   :file: lazytable.csv
   :header-rows: 1

Discussion
==========

Regarding the experiments with the lazy learning approach, we see, that the feature selection has less impact on the prediction results.
While the accuracy without any feature selection is 67.20%, the accuracy of the other methods is equally or worse.
Those results differ from the results in the paper, which are given in brackets. They claim to archieve better results using the feature selection.

The further consideration of the rather similar accuracy values we have obtained suggests, that the Naive Bayes constantly predicts the same value.
We verify this assumption with the calculation of the recall and precision.
Since the recall of the postive class is nearly 0, the Naive Bayes is not learning.
Computing the proportion of positive and negative occurences, we get a value near the precision scores, so the estimator chosen in the papers does not fit to the used data set.
Especially for the methods HNB and RNB which allow to restrict the number of chosen feature resulting in very small compression rates, the Naive Bayes predicts athe negative class almost always.

Hence, we repeated the experiments with a Gaussian Naive Bayes and a Decision Tree, but obtained similar predictions seeming like the classifier has not learned from the features.

We assume that the presented feature selection approaches - filtering out a lot of data - may be more valuable in larger datasets.

.. [1] Downloaded from https://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/SportTweets/ (2nd July 2023)
