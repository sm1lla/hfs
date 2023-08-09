####################
MR and HIP
####################

2015 Wan and Freitas extended their previous paper with three further feature selection methods,
from which two are implemented in this library (the third one shows high similarity to HNB-s) :cite:p:`todo`.
The Hierarchical Information-Preserving (HIP) algorithm eliminates redundancy by selecting the deepest positive or the highest negative feature.
So for each instance and each feature :math:`f_i`, if the value of all :math:`f_i` equals 1,all ancestors :math:`f_a` are removed while if it 0 all descendants :math:`f_d` will be removed.
In contrast to their paper of 2013 this algorithm does not include the relevance score.
They close this gap with Most Relevant (MR) algorithm, which relies on the previously defined relevance measure.
MR selects the most relevant features on each path of the sets with a 1 as value and the sets with a 0.
After processing each path only the most relevant terms are left and any redundancy is eliminated.

As processing each path is hardly feasible for big hierarchical structures, the implementation traverses the nodes in topological order saving the most relevant nodes (mr's) seen up to this point :math:`f_i`.
In the case of :math:`f_i=1`, for each predecessor :math:`f_{i+1}` and each saved mr, if this has a higher relevance than :math:`f_i`,
the status of :math:`f_i` is set to removal and the mr is saved to :math:`f_i`.
If there is one mr, which relevance is lower, meaning on this path ending in :math:`f_i`, :math:`f_i` has the highest value, :math:`f_i` itself is saved in the mr's of its own node.
For :math:`f_i=0` the process is similar performed in the reverse topological order. More information can be found in the commented code itself.
In contrast to the implementation of HIP is self-explaining and rather straight-forward.
