import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def plotauc(name, f1_list, f1auc, flat):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    if flat == 'best':
        pdf = PdfPages(f'./plots/{name}/f1auc_best_output.pdf')
    else:
        pdf = PdfPages(f'./plots/{name}/f1auc_val_output.pdf')
    
    k = np.arange(0, 1, 0.01)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(k, f1_list, y2=0, alpha=.5, linewidth=0)
    ax.set_xlabel('K')
    ax.set_ylabel('F1')
    ax.set_title(f'f1auc = {f1auc}')
    plt.plot(k, f1_list)
    pdf.savefig(fig)
    plt.close()
    pdf.close()
    
    
# while testing
def plotTwoLoss(name, y_true, pred12, pred1, loss12, loss1, labels,predict_label,result,booltrain):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/{booltrain}_output.pdf')
    for dim in range(y_true.shape[1]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,2),sharex=True)
        ax1.set_ylabel('Magnorm')
        ax1.set_title(f'Dimension = {dim}')
        
        ax1.plot(pred1[:, dim], '-', linewidth=0.01, color='b', label='pred1')
        ax1.plot(pred12[:, dim], '-', linewidth=0.01, color='r', label='pred12')
        ax1.plot(y_true[:, dim], '-', linewidth=0.1, color='k',label='ground truth')
        
        if 'test' in booltrain:
            ax1.plot(labels[:, dim], '--', linewidth=0.1, alpha=0.8, color='blue')
            ax1.fill_between(np.arange(labels[:, dim].shape[0]), 0, labels[:, dim], color='blue', alpha=0.3)
            ax1.plot(predict_label[dim,:], '--', linewidth=0.1, alpha=0.8, color='y')
            ax1.fill_between(np.arange(predict_label.shape[-1]), 0, predict_label[dim,:], color='y', alpha=0.3)
        
        
        ax2.plot(loss1[:, dim], '-', linewidth=0.1, color='b', label='loss1')
        ax2.plot(loss12[:, dim], '-',linewidth=0.1, color='r',label='loss12')
        
        th2 = result['threshold']
        th_list2 = [th2] * len(loss12)

        ax2.plot(th_list2, '--',linewidth=0.8, color='k', label='threshold')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('MSELoss')

        if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        
        pdf.savefig(fig)
        plt.close()
    pdf.close()
    print("plot end")

def plotOneLoss(name, y_true, pred1, loss1, labels, predict_label, result, booltrain=False):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/{booltrain}_output.pdf')
    for dim in range(y_true.shape[1]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 2), sharex=True)
        ax1.set_ylabel('Magnorm')
        ax1.set_title(f'Dimension = {dim}')
        ax1.plot(pred1[:, dim], '-', linewidth=0.01, color='b', label='pred1')
        ax1.plot(y_true[:, dim], '-', linewidth=0.1, color='k', label='ground truth')
        
        if booltrain == 'test':
            ax1.plot(labels[:, dim], '--', linewidth=0.1, alpha=0.8, color='blue')
            ax1.fill_between(np.arange(labels[:, dim].shape[0]), 0, labels[:, dim], color='blue', alpha=0.3)
            ax1.plot(predict_label[dim, :], '--', linewidth=0.1, alpha=0.8, color='y')
            ax1.fill_between(np.arange(predict_label.shape[-1]), 0, predict_label[dim, :], color='y', alpha=0.3)
    
        ax2.plot(loss1[:, dim], '-', linewidth=0.1, color='b', label='loss1')
        th2 = result['threshold']
        th_list2 = [th2] * predict_label.shape[-1]
        ax2.plot(th_list2, '--', linewidth=0.5, color='k', label='threshold')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('MSELoss')
        if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        pdf.savefig(fig)
        plt.close()
    pdf.close()