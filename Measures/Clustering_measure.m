% 采用六种度量指标度量聚类精度
[ARI,NMI,ACC] = Clustering_measure(label,gt);
% 输入：label: your labels
% gt: the real labels

% 输出：六个聚类的评价指标
addpath 'Measures'
ARI = RandIndex(label,gt);
NMI_1 = MutualInfo(label,gt);
ACC =  Accuracy(label,gt);