Setup
######

For the purpose of evaluating the implemented methods we use two different data sets, following the mentioned papers to lazy and eager learning. 
While the first one suggests a bioinformatic data set, the latter one uses [todo-smilla].

Bioinformatic data set
**********************

For the purpose of evaluating the lazy methods we follow Wan, Freitas and de Magalhaes predicting a *C. elegans* gene’s effect on the organism’s longevity. 
The corresponding information is obtained from the database of Human Ageing Genomic Resources (:cite:p:`TacutuRobi2013HAGR`).
Hierarchical relations are pictured in the Gene Ontology (GO) (:cite:p:`GeneOntologyConsortium2004`) from the National Institutes of Health. 

The GO collects terms and hierarchical relationships in a "is-a" generelizations/specializations, among those, building a directed acyclic graph.
If a gene is associated with the GO term i, it is also associated with all ancestors of i in the hierarchy (assured in the preprocessing of instances).
The gene2go :cite:p:`gene2go` maps the GO-terms to the associated genes.
We build our data set by selecting all EntrezID numbers of each *C. elegans* gene in the HAGR database with the same EntrezID number in the NCBI gene database gene2go [2023-06-25]. 
Then the GO terms for matched genes are mapped to the data from the gene2go database. Thus, we obtain a data set with 889 genes (rows) and 27416 go-terms (columns) set to 1 if the go-term is present in this gene.
After deleting genes associated with no go-terms the data set includes 819 genes and the final set with only genes which effect on longevity is known, the data set contains 793 rows.
