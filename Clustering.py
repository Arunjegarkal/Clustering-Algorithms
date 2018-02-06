#%matplotlib inline

import matplotlib
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time


col1 = []
col2 = []

count=0
with open("Clustering-dataset/1.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        #print line
        col1.append(float(line[0]))
        col2.append(float(line[1]))
        count=count+1


import csv
data = list(csv.reader(open('Clustering-dataset/1.txt'), dialect="excel-tab"))
#print(data)

#print col1
#for n in col1:
#    print n

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
plt.scatter(col1,col2, c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)
plt.title('Ploting Datapoints', fontsize=24)
#plt.text(9, 33, 'Ploting Datapoints', fontsize=20)
plt.xlabel("Column 1", fontsize=14)
plt.ylabel("Column 2", fontsize=14)
plt.show()

def plot_clusters(col1,col2,data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (1, 1, 1) for x in labels]
    #colors = ['r','b','y','g','c','m']
    plt.scatter(col1, col2, c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=18)
    #plt.text(28, -3, 'Performance: Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.text(28, -3, ''.format(end_time - start_time), fontsize=14)
    plt.xlabel("Column 1", fontsize=14)
    plt.ylabel("Column 2", fontsize=14)
    plt.show()
    

plot_clusters(col1,col2,data, cluster.KMeans, (), {'n_clusters':2})
plot_clusters(col1,col2,data, cluster.KMeans, (), {'n_clusters':4})
plot_clusters(col1,col2,data, cluster.KMeans, (), {'n_clusters':6})
plot_clusters(col1,col2,data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'average'})
plot_clusters(col1,col2,data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'complete'})
plot_clusters(col1,col2,data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})
#plot_clusters(col1,col2,data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})
#plot_clusters(col1,col2,data,  cluster.SpectralClustering, (), {'n_clusters':6})
#plot_clusters(col1,col2,data, cluster.AgglomerativeClustering, (), {'n_clusters':4, 'linkage':'average'})
plot_clusters(col1,col2,data, cluster.DBSCAN, (), {'eps':1})
plot_clusters(col1,col2,data, cluster.DBSCAN, (), {'eps':2})
plot_clusters(col1,col2,data, cluster.DBSCAN, (), {'eps':10})
print "done"

