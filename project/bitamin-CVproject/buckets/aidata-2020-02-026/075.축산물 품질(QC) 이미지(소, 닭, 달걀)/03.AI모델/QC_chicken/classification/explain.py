import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, multilabel_confusion_matrix


# Confusion Table & Accuracy
class Prediction:
    def __init__(self, total_recon_loss, true_label_tot, threshold):
        self.total_recon_loss=total_recon_loss
        self.true_label_tot=true_label_tot
        self.mean=np.mean(total_recon_loss)
        self.std=np.std(total_recon_loss)
        
        self.val_best_f1_threshold=threshold
                
    def single_threshold(self, threshold):
        
        # pred_label_tot
        self.pred_label_tot=[]
        for error in self.total_recon_loss:
            if (threshold<error):
                self.pred_label_tot.append(1)
            else:
                self.pred_label_tot.append(0)     
        
        
    def cal_for_curve(self, threshold):        
        self.single_threshold(threshold)
        f1=f1_score(self.true_label_tot, self.pred_label_tot, average='macro')       
                
        return f1
    
    
    def get_prediction(self, test=True): 
        
        max_loss=max(self.total_recon_loss)
        min_loss=min(self.total_recon_loss)
        step=(max_loss-min_loss)/700
        self.threshold=[min_loss+i*step for i in range(1,700)]        

        self.total_f1_list=[]
        
        for num in self.threshold:            
            f1=self.cal_for_curve(num)       
            self.total_f1_list.append(f1)
            
        # best f1 score
        self.max_f1=max(self.total_f1_list)
        max_idx=self.total_f1_list.index(self.max_f1)
        self.best_f1_threshold=self.threshold[max_idx]
        
        # Final result
        self.single_threshold(self.best_f1_threshold)
        if test:
            self.single_threshold(self.val_best_f1_threshold)
        
        return self.true_label_tot, self.pred_label_tot
        

        



class ShowResult:
    def __init__(self, true_tot_labels, pred_tot_labels):
        self.true_tot_labels=true_tot_labels
        self.pred_tot_labels=pred_tot_labels
        self.label=['Normal','Abnormal']
        
        self.multi_label_confusion_mat=multilabel_confusion_matrix(self.true_tot_labels, self.pred_tot_labels)
        self.total_num=len(true_tot_labels)
        self.total_f1=f1_score(self.true_tot_labels, self.pred_tot_labels, average=None).tolist()
        
    def per_class_confusion_mat(self, array, label):
        index=pd.MultiIndex.from_arrays([ ['True','True'], [f'Non {label}', label] ])
        columns=pd.MultiIndex.from_arrays([ ['Pred','Pred'], [f'Non {label}', label] ])
        
        cf_mat=pd.DataFrame(array, index=index, columns=columns)
        
        print(f'#-- Confusion Matrix for class {label}\n')
        print(cf_mat)    
        
        print(f"F1-Score for class {label} : {self.total_f1[self.label.index(label)] :.3f}")
        print('-'*35)
        print()
        
        
        
    def show_result(self):
        cf_mat=pd.crosstab(pd.Series(self.true_tot_labels), pd.Series(self.pred_tot_labels),
                               rownames=['True'], colnames=['Predicted'], margins=True)
        cf_mat=cf_mat.rename(index={0:'Normal', 1:'Abnormal'},
                      columns={0:'Normal', 1:'Abnormal'})

        print(cf_mat)
        print()
        print()       
        
        self.total_acc=[]
        for i, label in enumerate(self.label):
            array=self.multi_label_confusion_mat[i]
            self.per_class_confusion_mat(array, label)

            
        print(f"#-- Final Macro F1-Score")
        print(f"( {self.total_f1[0] :.3f} + {self.total_f1[1] :.3f} ) / 2 = {np.mean(self.total_f1) :.4f}")