# run d

from itertools import cycle
import pandas
from numpy import interp
import numpy
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

tik = time.clock()

df = pandas.read_csv('real_diamond.csv')

x = df.iloc[:,0:8].values
Y = df.iloc[:,8].values

# untuk ROC
y = label_binarize(Y, classes=['fair','good','very good','premium','ideal'])
n_classes = y.shape[1]
'''
print 'y_label_binarize = '
print y.shape
print y, '\n'
'''

X_standarizing = StandardScaler().fit_transform(x)
'''
print 'X_standarizing = '
print X_standarizing.shape
print X_standarizing, '\n'
'''
pca = PCA().fit(X_standarizing)
pca_variance_ratio = pca.explained_variance_ratio_

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(8), pca_variance_ratio, alpha=0.5, align='center',
            label='individual explained variance')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.savefig('PREDI2.png', format='png', dpi=250)
plt.show()#55.314   14.404


'''
print 'explained variance ratio = '
print pca_variance_ratio.shape
print pca_variance_ratio, '\n'
'''

'''
plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,8,1)
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.savefig('RF_PREDI3.png', format='png', dpi=1200)
plt.show()
'''


sklearn_pca = PCA(n_components=5)
x_transform = sklearn_pca.fit_transform(X_standarizing)


random_forest = RandomForestClassifier(n_estimators=10,
                                       criterion='gini',
                                       max_depth=None,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0,
                                       max_features=None,
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       min_impurity_split=None,
                                       bootstrap=True,
                                       oob_score=False,
                                       n_jobs=1,
                                       random_state=None,
                                       verbose=0,
                                       warm_start=False,
                                       class_weight=None
									   )




# untuk ROC
x_train, x_test, y_train, y_test = train_test_split(x_transform, y, train_size = 0.90, test_size = 0.10, random_state = 0)

#ovr = OneVsRestClassifier(random_forest)
y_score = random_forest.fit(x_train, y_train).predict(x_test)
#y_score = ovr.fit(x_transform, y_score).predict(x_transform)


'''
print roc_curve(y_test[:, 0], y_score[:, 0])
print roc_curve(y_test[:, 1], y_score[:, 1])
print roc_curve(y_test[:, 2], y_score[:, 2])
print roc_curve(y_test[:, 3], y_score[:, 3])
print roc_curve(y_test[:, 4], y_score[:, 4])
'''
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
	
# Compute micro-average ROC curve and ROC area
'''
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
'''

print 'tp rate = ', tpr[0],tpr[1],tpr[2],tpr[3],tpr[4]
print 'fp rate = ', fpr[0],fpr[1],fpr[2],fpr[3],fpr[4]
#print 'roc curve = ', roc_curve[0],roc_curve[1],roc_curve[2],roc_curve[3],roc_curve[4]
print 'auc = ', roc_auc[0], roc_auc[1], roc_auc[2], roc_auc[3], roc_auc[4]

# Plot ROC curve
'''
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = numpy.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
'''

# Plot all ROC curves
plt.figure()
'''
plt.plot(fpr["micro"], tpr["micro"],
     label='micro-average ROC curve (area = {0:0.2f})'
           ''.format(roc_auc["micro"]),
     color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
     label='macro-average ROC curve (area = {0:0.2f})'
           ''.format(roc_auc["macro"]),
     color='navy', linestyle=':', linewidth=4)
'''
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1, 
         label='ROC curve of class {0} (area = {1:0.2f})'
         ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], linestyle = '--', lw=2, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('ROC_CURVE.png', format='png', dpi=250)
plt.show()


tok = time.clock()

time_total = tok-tik
print 'total waktu program = ',  time_total


