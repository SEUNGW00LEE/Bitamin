import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score

class ShowResult:
    def __init__(self, true_tot_labels, pred_tot_labels):
        self.true_tot_labels=true_tot_labels
        self.pred_tot_labels=pred_tot_labels
        self.label=['A+','A']
        
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
        cf_mat=cf_mat.rename(index={0:'A+', 1:'A'},
                      columns={0:'A+', 1:'A'})

        print(cf_mat)
        print()
        print()       
        
        self.total_acc=[]
        for i, label in enumerate(self.label):
            array=self.multi_label_confusion_mat[i]
            self.per_class_confusion_mat(array, label)

            
        print(f"#-- Final Macro F1-Score")
        print(f"( {self.total_f1[0] :.3f} + {self.total_f1[1] :.3f} ) / 2 = {np.mean(self.total_f1) :.4f}")