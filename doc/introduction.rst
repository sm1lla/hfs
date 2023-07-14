Feature selection is a typical pre-processing step in machine learning. Those methods aim to improve the predictive performance \cite{fs}, for example by avoiding overfitting caused by noisy features. Feature selection thus may help to add generalisation.
Further a smaller training set can accelerate the training time, as the data the model has to include in its calculations is less.
Feature selection methods can increase the explainability of methods since users can concentrate on the most important variables and their meaning for the model.


Many real-world settings contain hierarchical relations between features. In text mining, words can be ordered in generalisation-specialisation relationships [todo-smilla add reference from papers?], while in Bioinformatics the function of genes is often described as hierarchy as for example in the go-term ontology, which collects terms in a directed acyclic graph \cite{go}.

While there are implementations of hierarchical classification methods \cite{Hiclass}, there are not many implementations [to do: find some] of feature selection methods that deal with hierarchical structures. 
Most methods assume flat feature spaces without hierarchical dependencies. But as redundancy and relevance of features can be obtained out of hierarchical information more precise \cite{wanbook}, hierarchical feature selection aims at adding more information to improve the step of selection. Hence, we propose a library, that implements methods presented by Wan and Freitas \cite{} and [Smilla].
 