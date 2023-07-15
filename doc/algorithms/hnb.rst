
HNB, HNB-s, RNB
================

2013 Wan, Freitas and de Magalhaes introduced an algorithm to select features based on their relevance and hierarchical redundancy \cite{hnb}.
The relevance is calculated by
 \begin{equation}\label{eq:1}
  \begin{aligned}
Relevance(\text{feature}) = (P(\text{Class}  = 1 \mid \text{feature} = 1) - P (\text{Class}  = 1 \mid \text{feature} = 0))^2 \\
+(P (\text{Class} = 0 | \text{feature} = 1) - P (\text{Class} = 0 | \text{feature} = 0))^2 
 \end{aligned}
\end{equation}\\

The Hierarchy Based Redundant Attribute Removal Naive Bayes Classifier (HNB) contains two phases which are executed for each test instance.
First, it considers every feature $f_i$. If its value is 1 it removes all ancestors $f_a$ ($f_a$ is an ancestor of $f_i$ iff $f_i$ is reachable from $f_a$), whose relevance is lower or equal than the relevance of $f_i$. Else if the value of $f_i$ is 0 it removes all descendants $f_d$ (reachable nodes from $f_i$) whose relevance is lower.
In a second step it selects the $k$ most relevant features from the obtained set.
On the selected features of the training set an estimator can now calculate a prediction for the testing instance. Wan, Freitas and de Magalhaes suggest the usage of an Naive Bayes Classifier, the library allows the combination with any sklearn-compatible binary classifier.
\\

The two phases can be executed separately from each other. The so called HNB-s selects all non-redundant features without performing any after-selecting steps. In contrast the Relevance-based Naive Bayes (RNB) only selects the top-k-ranked features in descending order of their individual predictive power measured by their relevance \ref{eq:1}. 
