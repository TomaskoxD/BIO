import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

class MetricCalculator:
    def __init__(self, df, predictions, target_names):
        self.df = df
        self.predictions = predictions
        self.target_names = target_names
        self.n_class = len(target_names)
        self.argmaxPRED = np.argmax(predictions, axis=1)
        self.metrics = None

    def calculate_metrics(self):
        Accuracy_score = accuracy_score(self.df, self.argmaxPRED)
        metrics = {
            'class_name': self.target_names,
            'F1': [],
            'AUC': 0,
            'Accuracy': Accuracy_score,
            'Sensitivity': [],
            'Precision': [],
            'Specificity': [],
            'ROC_curve': {},
            'tp': [],
            'tn': [],
            'fp': [],
            'fn': [],
        }

        for i in range(self.n_class):
            tmp_label = self.df == i
            tmp_pred = self.argmaxPRED == i
            F1 = f1_score(tmp_label, tmp_pred)
            tn, fp, fn, tp = confusion_matrix(tmp_label, tmp_pred).ravel()
            outAUROC = roc_auc_score(tmp_label, self.predictions[:, i])

            metrics['F1'].append(F1)
            metrics['tp'].append(tp)
            metrics['tn'].append(tn)
            metrics['fp'].append(fp)
            metrics['fn'].append(fn)

            metrics['ROC_curve']['ROC_fpr_' + str(i)] = roc_curve(tmp_label, self.predictions[:, i])[0]
            metrics['ROC_curve']['ROC_tpr_' + str(i)] = roc_curve(tmp_label, self.predictions[:, i])[1]
            metrics['ROC_curve']['ROC_T_' + str(i)] = roc_curve(tmp_label, self.predictions[:, i])[2]
            metrics['ROC_curve']['AUC_' + str(i)] = outAUROC

            metrics['AUC'] += outAUROC
            metrics['Sensitivity'].append(tp / (tp + fn))
            metrics['Specificity'].append(tn / (fp + tn))
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            metrics['Precision'].append(precision)

        mPrecision = sum(metrics['tp']) / sum(metrics['tp'] + metrics['fp'])
        mRecall = sum(metrics['tp']) / sum(metrics['tp'] + metrics['fn'])

        metrics['micro-Precision'] = mPrecision
        metrics['micro-Sensitivity'] = mRecall
        metrics['micro-Specificity'] = sum(metrics['tn']) / sum(metrics['fp'] + metrics['tn'])
        metrics['micro-F1'] = 2 * mPrecision * mRecall / (mPrecision + mRecall)
        metrics['AUC'] /= self.n_class

        self.metrics = metrics

    def get_metrics(self):
        if self.metrics is None:
            self.calculate_metrics()
        return self.metrics

    def print_metrics(self, model, clasified_images_dir):
        
        print('\nValues classificated by model: ' + model + ' saved in ' + clasified_images_dir  + ' have following metrics: \n')

        print('Accuracy: ' + str("{:0.4f}".format(np.mean(self.metrics['Accuracy']))))
        print('F1: ' + str("{:0.4f}".format(np.mean(self.metrics['F1']))))
        print('AUC: ' + str("{:0.4f}".format(np.mean(self.metrics['AUC']))))
        print('Sensitivity: ' + str("{:0.4f}".format(np.mean(self.metrics['Sensitivity']))))
        print('Precision: ' + str("{:0.4f}".format(np.mean(self.metrics['Precision']))))
        print('Specificity: ' + str("{:0.4f}".format(np.mean(self.metrics['Specificity']))))
        print('micro-Precision: ' + str("{:0.4f}".format(np.mean(self.metrics['micro-Precision']))))
        print('micro-Sensitivity: ' + str("{:0.4f}".format(np.mean(self.metrics['micro-Sensitivity']))))
        print('micro-Specificity: ' + str("{:0.4f}".format(np.mean(self.metrics['micro-Specificity']))))
        print('micro-F1: ' + str("{:0.4f}".format(np.mean(self.metrics['micro-F1']))))
        print('tp: ' + str("{:0.4f}".format(np.mean(self.metrics['tp']))))
        print('tn: ' + str("{:0.4f}".format(np.mean(self.metrics['tn']))))
        print('fp: ' + str("{:0.4f}".format(np.mean(self.metrics['fp']))))
        print('fn: ' + str("{:0.4f}".format(np.mean(self.metrics['fn']))))