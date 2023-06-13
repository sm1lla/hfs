
import pandas as pd
import numpy as np



from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import TreeSearch, BayesianEstimator



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from test.test_fixtures import *

def get_bayesian_network(dag, data):

    model = BayesianModel(dag)
    model.fit()