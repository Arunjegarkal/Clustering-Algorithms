library(ggplot2)
library(readtext)
library("dbscan")

dataset1<-read.delim("Clustering-dataset//Clustering2.txt",sep = '\t')

data<-dataset1[,-3]
colnames(data)<-c("column1","column2")

ggplot(data[, 1:2], aes(column1,column2)) + geom_point()

km<-kmeans(data, 6, nstart = 20)

km$cluster <- as.factor(km$cluster)
ggplot(data[, 1:2], aes(column1,column2, color = km$cluster)) + geom_point()

#Heirachical Clustering

clusters <- hclust(dist(data))
plot(clusters)
clusterCut <- cutree(clusters, 3)
plot(clusterCut)

clusters <- hclust(dist(data), method = 'complete')
plot(clusters)

#DBScan
x <- as.matrix(data)
db <- dbscan(x, eps = 1, minPts = 4)

pairs(x, col = db$cluster + 1L)
