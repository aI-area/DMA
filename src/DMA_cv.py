# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split,StratifiedKFold
import matplotlib.pyplot as plt
from utils import *
from tflearn.activations import relu
from optparse import OptionParser
from scipy import interp

parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k")
parser.add_option("-t", "--t", default="o", help="Test scenario")
(opts, args) = parser.parse_args()

def loadtxt(path):
    data = np.loadtxt(path,dtype=str)

    DATA = []
    for i in data:
        DATA.append(i.split(','))

    return np.array(DATA,dtype=float)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

# load network
network_path = '../data/'
save_path = '../result/'
mirna_name = np.loadtxt(network_path + 'miRNA-name.txt', dtype=str)
disease_similarity_matrix = loadtxt(network_path + 'DSmat.txt')
mirna_similarity_matrix = loadtxt(network_path + 'RSmat.txt')
mirna_disease_associations = loadtxt(network_path + 'RDmat.txt')

DMA = mirna_disease_associations.T
DS = row_normalize(disease_similarity_matrix, True)
MS = row_normalize(mirna_similarity_matrix, True)

[num_disease, num_mirna] = DMA.shape
dim_disease = int(opts.d)
dim_mirna = int(opts.d)
dim_pred = int(opts.k)
dim_pass = int(opts.d)

class Model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.disease_disease = tf.placeholder(tf.float32, [num_disease, num_disease])
        self.disease_disease_normalize = tf.placeholder(tf.float32, [num_disease, num_disease])
        self.mirna_mirna = tf.placeholder(tf.float32, [num_mirna, num_mirna])
        self.mirna_mirna_normalize = tf.placeholder(tf.float32, [num_mirna, num_mirna])
        self.mirna_disease = tf.placeholder(tf.float32, [num_mirna, num_disease])
        self.mirna_disease_normalize = tf.placeholder(tf.float32, [num_mirna, num_disease])
        self.disease_mirna = tf.placeholder(tf.float32, [num_disease, num_mirna])
        self.disease_mirna_normalize = tf.placeholder(tf.float32, [num_disease, num_mirna])
        self.disease_mirna_mask = tf.placeholder(tf.float32, [num_disease + num_mirna, num_disease + num_mirna])

        self.A = tf.concat([tf.concat([self.disease_disease, self.disease_mirna], 1),
                            tf.concat([self.mirna_disease, self.mirna_mirna], 1)], 0)

        self.disease_embedding = weight_variable([num_disease, dim_disease])
        self.mirna_embedding = weight_variable([num_mirna, dim_mirna])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.disease_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.mirna_embedding))

        W0 = weight_variable([dim_pass + dim_disease, dim_disease])
        b0 = bias_variable([dim_disease])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        disease_vector0 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.disease_mirna_normalize,
                                 a_layer(self.mirna_embedding, dim_pass)) +
                       tf.matmul(self.disease_disease_normalize,
                                 a_layer(self.disease_embedding, dim_pass)),
                       self.disease_embedding], axis=1), W0) + b0), dim=1)

        mirna_vector0 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.mirna_mirna_normalize,
                                 a_layer(self.mirna_embedding, dim_pass)) +
                       tf.matmul(self.mirna_disease_normalize,
                                 a_layer(self.disease_embedding, dim_pass)),
                       self.mirna_embedding], axis=1), W0) + b0), dim=1)

        self.disease_representation = disease_vector0
        self.mirna_representation = mirna_vector0
        self.features_matrix = tf.concat([self.disease_representation, self.mirna_representation], 0)
        self.A_reconstruct = bi_layer(self.features_matrix, self.features_matrix, sym=True, dim_pred=dim_pred)

        tmp = tf.multiply(self.disease_mirna_mask, (self.A_reconstruct - self.A))
        self.A_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))
        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))
        self.loss = self.A_reconstruct_loss + self.l2_loss


graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    DR_pred = model.A_reconstruct[:num_disease, num_disease:]
    RD_pred = model.A_reconstruct[num_disease:, :num_disease]
    eval_pred = (DR_pred + tf.transpose(RD_pred, perm=[1,0])) / 2.0

def train_and_evaluate(DMAtrain, DMAtest, DMAcandidate, graph, verbose=True, num_steps = 2000):
    mask = np.zeros((num_disease,num_mirna))
    disease_mirna = np.zeros((num_disease,num_mirna))
    for ele in DMAtrain:
        disease_mirna[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
    mirna_disease = disease_mirna.T
    disease_mirna_normalize = row_normalize(disease_mirna,False)
    mirna_disease_normalize = row_normalize(mirna_disease,False)
    mask = np.concatenate([np.concatenate([np.ones((num_disease, num_disease)), mask], axis=1),
                           np.concatenate([mask.T, np.ones((num_mirna, num_mirna))], axis=1)], axis=0)


    lr = 0.0005
    scores = []
    labels = []
    min_loss = float('inf')
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in range(num_steps):
            _, tloss, results = sess.run([optimizer,total_loss,eval_pred],
                                        feed_dict={model.disease_mirna:disease_mirna, model.disease_mirna_normalize:disease_mirna_normalize,
                                        model.mirna_disease:mirna_disease, model.mirna_disease_normalize:mirna_disease_normalize,
                                        model.disease_disease:disease_similarity_matrix, model.mirna_mirna:mirna_similarity_matrix,
                                        model.disease_disease_normalize:DS, model.mirna_mirna_normalize:MS,
                                        model.disease_mirna_mask:mask, learning_rate: lr})
            #every 20 steps of gradient descent, evaluate the performance, other choices of this number are possible
            if i % 20 == 0 and verbose == True:
                print('step',i,'total loss',tloss)

                if tloss <= min_loss:
                    min_loss = tloss
                    best_results = results


        for ele in DMAtest:
            scores.append(best_results[ele[0], ele[1]])
            labels.append(ele[2])
        for ele in DMAcandidate:
            scores.append(best_results[ele[0], ele[1]])
            labels.append(ele[2])

    return scores, labels

def divide_known_unknown_associations(A, exception=None, special=None):
    known = []
    unknown = []
    if special != None:
        for j in range(A.shape[1]):
            if A[special][j] == 1:
                known.append([special,j,1])
            else:
                unknown.append([special,j,0])
    else:
        for i in range(A.shape[0]):
            if i == exception: pass
            for j in range(A.shape[1]):
                if A[i][j] == 1:
                    known.append([i,j,1])
                else:
                    unknown.append([i,j,0])

    return np.array(known), np.array(unknown)

def plot_roc_curve(fpr, tpr, auc):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.plot(fpr, tpr, color='b', label=r'Mean ROC (AUC = %0.6f)' % auc, lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='best')
    plt.show()

def plot_pr_curve(recall, precision, aupr):
    plt.figure(2)
    plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.plot(recall, precision, color='b', label=r'Mean PR (AUPR = %0.6f)' % aupr, lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def FCV(epochs=2000):
    known_associations, unknown_associations = divide_known_unknown_associations(DMA)

    rs = np.random.randint(0, 1000, 1)[0]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    kf = kf.split(known_associations[:,:-1], known_associations[:,-1])

    tpr_fold = []
    precision_fold = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    DMAcandidate = unknown_associations
    for train_idx, test_idx in kf:
        DMAtrain = known_associations[train_idx]
        DMAtest = known_associations[test_idx]
        """
        # used to adjust the superparameter
        DMAtrain, DMAvalid = train_test_split(DMAtrain, test_size=0.05, random_state=rs)
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAvalid, DMAcandidate=DMAcandidate, 
                                            graph=graph, num_steps=epochs)
        """
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAtest, DMAcandidate=DMAcandidate,
                                            graph=graph, num_steps=epochs)

        fpr, tpr, _ = roc_curve(labels, scores)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_fold.append(interp_tpr)

        precision, recall, _ = precision_recall_curve(labels, scores)
        rank_idx = np.argsort(recall)
        recall = recall[rank_idx]
        precision = precision[rank_idx]
        interp_precision = interp(mean_recall, recall, precision)
        interp_precision[0] = 1.0
        precision_fold.append(interp_precision)

    mean_tpr = np.mean(tpr_fold, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #plot_roc_curve(mean_fpr, mean_tpr, mean_auc)
    roc = pd.DataFrame({'fpr':mean_fpr,'tpr':mean_tpr})
    roc.to_csv(save_path + '5fold_roc.csv', index=False)

    mean_precision = np.mean(precision_fold, axis=0)
    mean_precision[-1] = 0.0
    mean_aupr = auc(mean_recall, mean_precision)
    #plot_pr_curve(mean_recall, mean_precision, mean_aupr)
    pr = pd.DataFrame({'recall': mean_recall, 'precision': mean_precision})
    pr.to_csv(save_path + '5fold_pr.csv', index=False)

    print('5-fold auc %0.6f', mean_auc, ', 5-fold aupr %0.6f', mean_aupr)


def LODOCV(epochs=2000):
    tpr_fold = []
    precision_fold = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    for i in range(num_disease):
        print('current cv: %d, total cv: %d' % (i+1, num_disease))

        DMAtrain, _ = divide_known_unknown_associations(DMA, exception=i)
        DMAtest, DMAcandidate = divide_known_unknown_associations(DMA, special=i)
        """
        # used to adjust the superparameter
        rs = np.random.randint(0, 1000, 1)[0]
        DMAtrain, DMAvalid = train_test_split(DMAtrain, test_size=0.05, random_state=rs)
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAvalid, DMAcandidate=DMAcandidate, 
                                            graph=graph, num_steps=epochs)
        """
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAtest, DMAcandidate=DMAcandidate,
                                            graph=graph, num_steps=epochs)

        fpr, tpr, _ = roc_curve(labels, scores)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_fold.append(interp_tpr)

        precision, recall, _ = precision_recall_curve(labels, scores)
        rank_idx = np.argsort(recall)
        recall = recall[rank_idx]
        precision = precision[rank_idx]
        interp_precision = interp(mean_recall, recall, precision)
        interp_precision[0] = 1.0
        precision_fold.append(interp_precision)

    mean_tpr = np.mean(tpr_fold, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #plot_roc_curve(mean_fpr, mean_tpr, mean_auc)
    roc = pd.DataFrame({'fpr': mean_fpr, 'tpr': mean_tpr})
    roc.to_csv(save_path + 'lodocv_roc.csv', index=False)

    mean_precision = np.mean(precision_fold, axis=0)
    mean_precision[-1] = 0.0
    mean_aupr = auc(mean_recall, mean_precision)
    #plot_pr_curve(mean_recall, mean_precision, mean_aupr)
    pr = pd.DataFrame({'recall': mean_recall, 'precision': mean_precision})
    pr.to_csv(save_path + 'lodocv_pr.csv', index=False)

    print('lodocv auc %0.6f', mean_auc, ', lodocv aupr %0.6f', mean_aupr)

def GLOOCV(epochs):
    known_associations, unknown_associations = divide_known_unknown_associations(DMA)

    tpr_fold = []
    precision_fold = []
    n = len(known_associations)
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    DMAcandidate = unknown_associations
    for i in range(n):
        print('current cv: %d, total cv: %d' % (i+1, n))

        ele = known_associations[i]
        DMAtest = [ele]
        DMAtrain = np.delete(known_associations, i, axis=0)
        """
        # used to adjust the superparameter
        rs = np.random.randint(0, 1000, 1)[0]
        DMAtrain, DMAvalid = train_test_split(DMAtrain, test_size=0.05, random_state=rs)
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAvalid, DMAcandidate=DMAcandidate, 
                                            graph=graph, num_steps=epochs)
        """
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAtest, DMAcandidate=DMAcandidate,
                                            graph=graph, num_steps=epochs)

        fpr, tpr, _ = roc_curve(labels, scores)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_fold.append(interp_tpr)

        precision, recall, _ = precision_recall_curve(labels, scores)
        rank_idx = np.argsort(recall)
        recall = recall[rank_idx]
        precision = precision[rank_idx]
        interp_precision = interp(mean_recall, recall, precision)
        interp_precision[0] = 1.0
        precision_fold.append(interp_precision)

    mean_tpr = np.mean(tpr_fold, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #plot_roc_curve(mean_fpr, mean_tpr, mean_auc)
    roc = pd.DataFrame({'fpr': mean_fpr, 'tpr': mean_tpr})
    roc.to_csv(save_path + 'gloocv_roc.csv', index=False)

    mean_precision = np.mean(precision_fold, axis=0)
    mean_precision[-1] = 0.0
    mean_aupr = auc(mean_recall, mean_precision)
    #plot_pr_curve(mean_recall, mean_precision, mean_aupr)
    pr = pd.DataFrame({'recall': mean_recall, 'precision': mean_precision})
    pr.to_csv(save_path + 'gloocv_pr.csv', index=False)

    print('global loocv auc %0.6f' % mean_auc, ', global loocv aupr %0.6f' % mean_aupr)

def LLOOCV(epochs):
    known_associations, unknown_associations = divide_known_unknown_associations(DMA)

    tpr_fold = []
    precision_fold = []
    n = len(known_associations)
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    for i in range(n):
        print('current cv: %d, total cv: %d' % (i+1, n))

        ele = known_associations[i]
        DMAtest = [ele]
        DMAtrain = np.delete(known_associations, i, axis=0)
        _, DMAcandidate = divide_known_unknown_associations(DMA, special=ele[0])
        """
        # used to adjust the superparameter
        rs = np.random.randint(0, 1000, 1)[0]
        DMAtrain, DMAvalid = train_test_split(DMAtrain, test_size=0.05, random_state=rs)
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAvalid, DMAcandidate=DMAcandidate, 
                                            graph=graph, num_steps=epochs)
        """
        scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAtest, DMAcandidate=DMAcandidate,
                                            graph=graph, num_steps=epochs)

        fpr, tpr, _ = roc_curve(labels, scores)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_fold.append(interp_tpr)

        precision, recall, _ = precision_recall_curve(labels, scores)
        rank_idx = np.argsort(recall)
        recall = recall[rank_idx]
        precision = precision[rank_idx]
        interp_precision = interp(mean_recall, recall, precision)
        interp_precision[0] = 1.0
        precision_fold.append(interp_precision)

    mean_tpr = np.mean(tpr_fold, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #plot_roc_curve(mean_fpr, mean_tpr, mean_auc)
    roc = pd.DataFrame({'fpr': mean_fpr, 'tpr': mean_tpr})
    roc.to_csv(save_path + 'lloocv_roc.csv', index=False)

    mean_precision = np.mean(precision_fold, axis=0)
    mean_precision[-1] = 0.0
    mean_aupr = auc(mean_recall, mean_precision)
    #plot_pr_curve(mean_recall, mean_precision, mean_aupr)
    pr = pd.DataFrame({'recall': mean_recall, 'precision': mean_precision})
    pr.to_csv(save_path + 'lloocv_pr.csv', index=False)

    print('local loocv auc %0.6f' % mean_auc, ', local loocv aupr %0.6f' % mean_aupr)

def case_study1(epochs=2000):
    DMAtest = []
    DMAtrain, _ = divide_known_unknown_associations(DMA)
    _, DMAcandidate = divide_known_unknown_associations(DMA, special=288)
    scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAtest, DMAcandidate=DMAcandidate,
                                        graph=graph, num_steps=epochs)

    index = []
    for ele in DMAcandidate:
        index.append(ele[1])

    index = np.array(index)
    scores = np.array(scores)
    rank_idx = np.argsort(scores)
    index = index[rank_idx]
    index = index[-30:]

    for i in index:
        print(mirna_name[i])


def case_study2(epochs=2000):
    DMAtrain, _ = divide_known_unknown_associations(DMA, exception=271)
    DMAtest, DMAcandidate = divide_known_unknown_associations(DMA, special=271)
    scores, labels = train_and_evaluate(DMAtrain=DMAtrain, DMAtest=DMAtest, DMAcandidate=DMAcandidate,
                                        graph=graph, num_steps=epochs)

    index = []
    for ele in DMAtest:
        index.append(ele[1])
    for ele in DMAcandidate:
        index.append(ele[1])

    index = np.array(index)
    scores = np.array(scores)
    rank_idx = np.argsort(scores)
    index = index[rank_idx]
    index = index[-50:]

    for i in index:
        print(mirna_name[i])



if __name__ == '__main__':
    case_study2(2000)
