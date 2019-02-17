# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 08:44:15 2018

@author: Kashish
"""
##################################################################################################################################################
import nxviz

from IPython.display import SVG, display
import matplotlib.pyplot as plt
import csv
import json
import pandas as pd
import numpy as np
import sys
import json
import ast
import networkx as nx

import sys
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
#functions
def plot_network(G, node_size_dict, factor=10, **kwargs):
    nx.draw(
        G, 
        pos=nx.spring_layout(G),
        with_labels=True,
        node_size=[v*factor for v in dict_to_values(G, node_size_dict)],
        **kwargs
    )
    
def get_all_node_metrics(G):
    df = pd.DataFrame(index=G.nodes)
    df["degree"] = pd.Series(nx.degree_centrality(G))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G))
    df["closeness"] = pd.Series(nx.closeness_centrality(G))
    df["eigenvector"] = pd.Series(nx.eigenvector_centrality(G))
    df["clustering"] = pd.Series(nx.clustering(G))
    return df
def dict_to_values(G, dict_data):
    return [dict_data[n] for n in G.nodes()]
#functions for finding node similarity
def node_similarity(u, v, df):
    features = {}
    features["closeness"] = np.abs(all_node_metrics.loc[u, "closeness"] - all_node_metrics.loc[v, "closeness"])
    features["betweenness"] = np.abs(all_node_metrics.loc[u, "betweenness"] - all_node_metrics.loc[v, "betweenness"])
    features["eigenvector"] = np.abs(all_node_metrics.loc[u, "eigenvector"] - all_node_metrics.loc[v, "eigenvector"])
    features["clustering"] = np.abs(all_node_metrics.loc[u, "clustering"]- all_node_metrics.loc[v, "clustering"])
    features["degree"] = np.abs(all_node_metrics.loc[u, "degree"] -all_node_metrics.loc[v, "degree"])

    return features

#reading processed data
df=pd.read_csv("E://Spring 2018//Network Analysis//Project//network_data_cutoff_32.csv")

#removing isolates
df=df.loc[~(df["from"]==df["to"])]

#importing graph    
G = nx.from_pandas_edgelist(df, "Source", "Target")
len(G.nodes)
len(G.edges)
#plotting graph
nx.draw(G)
#calculating node degree
df_node_degree = pd.DataFrame([item for item in G.degree()])

degree_dict = G.degree
node_degrees = [
    degree_dict[n]*1
    for n in G.nodes
]
node_degrees
#plotting graph based on degree
nx.draw(G,node_size=node_degrees)
#calculating node centralities
all_node_metrics= get_all_node_metrics(G)

#finding  top nodes based on all centralities
df_betweenness=all_node_metrics.sort_values("betweenness", ascending=False)
df_closeness=all_node_metrics.sort_values("closeness", ascending=False)
df_eigenvector=all_node_metrics.sort_values("eigenvector", ascending=False)
df_clustering=all_node_metrics.sort_values("clustering", ascending=False)   
df_degree=all_node_metrics.sort_values("degree", ascending=False)

#plotting network based on different centrality value as node size
fig, ax = plt.subplots(1,1, figsize=(200,200))
plot_network(G,nx.closeness_centrality(G),factor=500)
plot_network(G,nx.betweenness_centrality(G),factor=5)
plot_network(G,nx.degree_centrality(G),factor=5)


# Applying community detection Algorithm

community=nx.algorithms.community.girvan_newman(G)

for nodes in nx.algorithms.community.girvan_newman(G):
    print(nodes)
top_level_communities = next(community)

#Louvain community detection algorithm
import community
part = community.best_partition(G)
values = [part.get(node) for node in G.nodes()]
fig, ax = plt.subplots(1,1, figsize=(100,100))
nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
# storing communities in the dictionary
com={}
for k,v in part.items(): com.setdefault(v,[]).append(k)

#Applying link prediction algorithm
data = []
for u in G.nodes:
    for v in G.nodes:
        if u!=v:
            features = node_similarity(u,v,df)
            features["u"] = u
            features["v"] = v
            features["is_edge"] = (u,v) in G.edges
            data.append(features)
linkpred=pd.DataFrame(data)
#counting class labels
linkpred.is_edge.value_counts()
#splitting predictors and predicted varible
X = linkpred[["closeness", "betweenness", "eigenvector", "degree","clustering"]]
Y = linkpred.is_edge
#applying cross validation

k_fold = KFold(len(X), n_folds=10, shuffle=True, random_state=0)
clf = RandomForestClassifier()
Score=cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)

#calculating confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
clf.fit(X_train,y_train)
predict=clf.predict(X_test)
#Recall score is also significant
print(classification_report(predict,y_test))
#confusion_matrix 
confusion_matrix(predict,y_test)
#feature importance

Feature_importance=pd.DataFrame([clf.feature_importances_], columns=X.columns)
############################################################################################################################################