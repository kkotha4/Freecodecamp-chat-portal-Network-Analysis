#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:42:38 2018

@author: ajit
"""

##################### DATA SOURCE #####################   
# https://medium.freecodecamp.org/we-just-released-3-years-of-freecodecamp-chat-history-as-open-data-all-5-million-messages-of-it-a03901f4d6fb



import os
os.chdir("/home/ajit/Documents/Network Analysis Project/data/all-posts-public-main-chatroom")

import csv
import json
import pandas as pd
import numpy as np
import sys
import ast
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

# read the input data line wise
csv.field_size_limit(sys.maxsize)
main_list = []
with open('freecodecamp_casual_chatroom.csv') as data:
    reader = csv.reader(data)
    i = 0
    for row in reader:
#        print(row)
        if row[17]!='[]':
            main_list.append([row[12], row[17]])
#            main_list.append([row[12], row[17], row[22]])
#            main_list.append([row[12], row[17], row[19]])
        '''
        i += 1
        if i > 1:
            break
           '''
main_list = main_list[1:len(main_list)]

# create data for one to one  from-to pairs of users
main_list2 = []
for i in range(len(main_list)):
    dict_list = ast.literal_eval(main_list[i][1])
    for j in range(len(dict_list)):
#       main_list2.append([main_list[i][0],dict(dict_list[j])['screenName'],main_list[i][2]])
        main_list2.append([main_list[i][0],dict(dict_list[j])['screenName']])        

main_df = pd.DataFrame(main_list2)        
        

# get frequency of all tuples of from,to names
count_dict = Counter(tuple(item) for item in main_list2)
density_list = []
size_list = []

# here we are finalizing which frequency is best for us, based on network density 
for freq_cutoff in range(70):
    count_dict_filter = {}
    print(freq_cutoff)    
# keep tuples which are above freq_cutoff
    for keys in count_dict.keys():
        if count_dict[keys]>freq_cutoff:
            count_dict_filter[keys] = count_dict[keys]
    
    count_dict_filter_2 = {}

# keep tuples which have also contain their inverted tuple    
    for keys in count_dict_filter.keys():
        new_key = [ keys[i] for i in [1,0]]
        #print(new_key)
        if tuple(new_key) in list(count_dict_filter.keys()):
            #print(new_key)
            count_dict_filter_2[keys] =(count_dict_filter[keys] , count_dict_filter[tuple(new_key)])
            #count_dict_filter_2[tuple(new_key)] = count_dict_filter[tuple(new_key)]
    
    main_df_2 = pd.DataFrame(list(count_dict_filter_2.keys()))
    G = nx.from_pandas_dataframe(main_df_2,0,1)
    density_list.append(nx.density(G))
    size_list.append(main_df_2.shape[0])
# end of for loop

#density_list = density_list0 + density_list 
#size_list = size_list0 + size_list

plt.plot(range(70),[1/i for i in density_list])
plt.plot(range(70),size_list)

#density_list0 = density_list 
#size_list0 = size_list

my_df = pd.DataFrame(density_list)
my_df.to_csv('final_density_list.csv', index=False, header=False)
my_df = pd.DataFrame(size_list)
my_df.to_csv('final_size_list.csv', index=False, header=False)

size_list[32 - 3]

################## FINAL Frequency Cutoff is 32 ################## 
######## Preparing the Final Data based on this Cutoff ########

######## the same code from above is repated ######## 
main_list = []
with open('freecodecamp_casual_chatroom.csv') as data:
    reader = csv.reader(data)
    i = 0
    for row in reader:
        if row[17]!='[]':
            main_list.append([row[12], row[17], row[19]])

main_list = main_list[1:len(main_list)]
main_list2 = []

for i in range(len(main_list)):
    dict_list = ast.literal_eval(main_list[i][1])
    for j in range(len(dict_list)):
        main_list2.append([main_list[i][0],dict(dict_list[j])['screenName'],main_list[i][2]])

main_df = pd.DataFrame(main_list2)

count_dict_filter = {}
freq_cutoff = 32
# keep tuples which are above freq_cutoff
for keys in count_dict.keys():
    if count_dict[keys]>freq_cutoff:
        count_dict_filter[keys] = count_dict[keys]

count_dict_filter_2 = {}
# keep tuples which have also contain their inverted tuple    
for keys in count_dict_filter.keys():
    new_key = [ keys[i] for i in [1,0]]
    #print(new_key)
    if tuple(new_key) in list(count_dict_filter.keys()):
        #print(new_key)
        count_dict_filter_2[keys] =(count_dict_filter[keys] , count_dict_filter[tuple(new_key)])

#for i in range(main_df_2.shape[0]):
'''
readby_sum_mean = []
cnt = 0
for i in count_dict_filter_2.keys():
    tmp= []
    for j in range(len(main_list2)):
        if [i[0],i[1]]==main_list2[j][0:1]:
            tmp.append(main_list2[j][2])
    readby_sum_mean.append([sum(tmp),np.mean(tmp)])
    cnt = cnt + 1
    if cnt%100==0: print(cnt)
'''

# save the resulting data
main_df_2 = pd.DataFrame({'from': [i[0] for i in list(count_dict_filter_2.keys())], 'to': [j[1] for j in list(count_dict_filter_2.keys())],
                                   'from_to_freq': [a[0] for a in count_dict_filter_2.values()], 'to_from_freq': [b[1] for b in count_dict_filter_2.values()] })
#                          'readby_sum': c[0] for c in readby_sum_mean, 'readby_mean': d[1] for d in readby_sum_mean})

main_df_2.to_csv('network_data_cutoff_32.csv',index=False, header=True)



###############################################################################
######################## Get the word frequencies in the messages #############

df=pd.read_csv("network_data_cutoff_32.csv")
df=df.loc[~(df["from"]==df["to"])]

lst = [i for i in df["from"]] + [j for j in df["to"]]
uniq_users =  [i for i in set(lst)]

main_df[2] = main_df[2].str.lower()
main_df2 = main_df[main_df[0].isin(uniq_users)] 
main_df2 = main_df[main_df[1].isin(uniq_users)] 


comm_dict = np.load("communities_best_partitions.npy")
comm_dict = comm_dict.all()

#comm = pd.DataFrame(data=comm_dict[1:,1:],    # values

for i in range(12):
    
    main_df3 = main_df2[main_df2[0].isin(comm_dict[i])]
    main_df3 = main_df3[main_df3[1].isin(comm_dict[i])]
    main_df3.columns = ["a","b","c"]
    
    cnt = main_df3.c.str.contains(r'welcome to fcc').sum()/main_df3.shape[0]
    
    #print(str(i) + str(" : ") + str(cnt))
    print(cnt)

