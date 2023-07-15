MR and HIP
===========

2015 Wan and Freitas extended their previous paper with three further feature selection methods, from which two are implemented in this library (the third one shows high similarity to HNB-s) \cite{TODO}.
The Hierarchical Information-Preserving (HIP) algorithm eliminates redundancy by selecting the deepest positive or the highest negative feature. So for each instance and each feature $f_i$, if the value of all  $f_i$ equals 1 all ancestors  $f_a$ are removed while if it 0 all descendants $f_d$ will be removed. In contrast to their paper of 2013 this algorithm does not include the relevance score.
They close this gap with Most Relevant (MR) algorithm, which relies on the previously defined relevance measure.
MR selects the most relevant features on each path of the sets with a 1 as value and the sets with a 0. After processing each path only the most relevant terms are left and any redundancy is eliminated.

As processing each path is hardly feasible for big hierarchical structures, the implementation traverses the nodes in topological order saving the most relevant nodes (mr's) seen up to this point $f_i$. In the case of $f_i = 1$, for each predecessor $f_{i+1}$ and each saved mr, if this has a higher relevance than $f_i$, the status of $f_i$ is set to removal and the mr is saved to $f_i$. If there is one mr, which relevance is lower, meaning on this path ending in $f_i$, $f_i$ has the highest value, $f_i$ itself is saved in the mr's of its own node. For $f_i=0$ the process is similar performed in the reverse topological order. More information can be find in the commented code itself.
In contrast the implementation of HIP is self-explaining and rather straight-forward.
