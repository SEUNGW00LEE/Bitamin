import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

class ShowResult:
    def __init__(self, true_tot_labels, pred_tot_labels):
        self.true_tot_labels=true_tot_labels
        self.pred_tot_labels=pred_tot_labels
        self.label=['1++','1+','1','2','3']
        
        self.multi_label_confusion_mat=multilabel_confusion_matrix(self.true_tot_labels, self.pred_tot_labels)
        self.total_num=len(true_tot_labels)
        
    def per_class_confusion_mat(self, array, label):
        index=pd.MultiIndex.from_arrays([ ['True','True'], [f'Non {label}', label] ])
        columns=pd.MultiIndex.from_arrays([ ['Pred','Pred'], [f'Non {label}', label] ])
        
        cf_mat=pd.DataFrame(array, index=index, columns=columns)
        acc=100*np.diag(cf_mat).sum()/self.total_num
        acc=round(acc, 3)        
        
        print(f'#-- Confusion Matrix for class {label}\n')
        print(cf_mat)
        print(f"\nAccuracy for class {label} : {acc}")
        print('-'*35)
        print()
        
        return acc
        
        
        
    def show_result(self):
        cf_mat=pd.crosstab(pd.Series(self.true_tot_labels), pd.Series(self.pred_tot_labels),
                               rownames=['True'], colnames=['Predicted'], margins=True)
        cf_mat=cf_mat.rename(index={0:'1++',1:'1+',2:'1',3:'2',4:'3'},
                      columns={0:'1++',1:'1+',2:'1',3:'2',4:'3'})

        print(cf_mat)
        print()
        print()       
        
        self.total_acc=[]
        for i, label in enumerate(self.label):
            array=self.multi_label_confusion_mat[i]
            acc=self.per_class_confusion_mat(array, label)
            self.total_acc.append(acc)
            
        print(f"#-- Final Average Accuracy")
        print(f"( {self.total_acc[0]} + {self.total_acc[1]} + {self.total_acc[2]} + {self.total_acc[3]} + {self.total_acc[4]} ) / 5 = {np.mean(self.total_acc) :.3f}")