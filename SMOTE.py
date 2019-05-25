from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix    
import xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import collections


row1=np.arange(1944)
results= pd.DataFrame(data=None,index=row1,columns = ['ratio', 'Class','Datasize','Training Datasize','After Sampling','Precision','Recall','f1-score','oobscore','Predicted Datasize','Testing Datasize','Result'])

data=pd.read_csv("sensor.csv")

data= data.drop(['timestamp','sensor_15'],axis=1)
data=data.fillna(data.mean())

target = data.loc[:, data.columns == 'machine_status']
print("original  data size",collections.Counter(target['machine_status']))
num_target=collections.Counter(target['machine_status'])
print("Noraml",num_target['NORMAL']/len(target))
print("Recovering",num_target['RECOVERING']/len(target))
print("Broken",num_target['BROKEN']/len(target))


X = data.loc[:, data.columns != 'machine_status']
y = data.loc[:, data.columns == 'machine_status']

#split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)




l1=[]
for i in np.arange(len(data)):
    l1.append(0)
data['Class']=l1

data.loc[data['machine_status']=='BROKEN', 'Class'] = 2

#data_train=data_train[data_train.xAttack!='u2r']
#data_train=data_train[data_train.xAttack!='dos']
#data_train=data_train[data_train.xAttack!='probe']
#data_train=data_train[data_train.xAttack!='r2l']
#data_train=data_train[data_train.xAttack!='normal']


y_train1 = data.loc[:, data.columns == 'machine_status']
#print("train data size",collections.Counter(y_train1['machine_status']))
datasize=collections.Counter(y_train1['machine_status'])
mapping1 = {'NORMAL': 1, 'RECOVERING': 2,'BROKEN' : 3}
data=data.replace({'machine_status': mapping1})
noclass_train1 = data.loc[:, data.columns != 'Class']
class_train1 = data.loc[:, data.columns == 'Class']







#data_test=data_test[data_test.xAttack!='u2r']
#data_test=data_test[data_test.xAttack!='dos']
#data_test=data_test[data_test.xAttack!='probe']
#data_test=data_test[data_test.xAttack!='r2l']
#data_test=data_test[data_test.xAttack!='normal']

#X_test = data_test.loc[:, data_test.columns != 'xAttack']
#y_test = data_test.loc[:, data_test.columns == 'xAttack']
#print("test data size",collections.Counter(y_test['xAttack']))




a1=0
A=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
B=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
C=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]

numofmaj1=datasize['NORMAL']+datasize['RECOVERING']
numofmin1=datasize['BROKEN']
noclass_train1_cols=noclass_train1.columns

for i1 in A:
    rat1=numofmin1/(numofmaj1*i1)

    
    if rat1<=1:
        sm1=SMOTE(sampling_strategy=rat1,random_state=5)
        
        noclass_train_sampled1,class_train_sampled1  = sm1.fit_sample(noclass_train1, class_train1.values.ravel())
        noclass_train_sampled1= pd.DataFrame(data=noclass_train_sampled1,columns =noclass_train1_cols )
        class_train_sampled1= pd.DataFrame(data=class_train_sampled1,columns =['Class'] )
        data_train2=pd.concat([noclass_train_sampled1, class_train_sampled1], axis=1)
        data_train2.loc[data_train2['machine_status']==3, 'Class'] = 0
        data_train2.loc[data_train2['machine_status']==2, 'Class'] = 1
        noclass_train2 = data_train2.loc[:, data_train2.columns != 'Class']
        class_train2 = data_train2.loc[:, data_train2.columns == 'Class']
        num_class_train2=collections.Counter(class_train2['Class'])
        numofmaj2=num_class_train2[0]
        numofmin2=num_class_train2[1]
        for i2 in B:
            rat2=numofmin2/(numofmaj2*i2)
            

            
            if rat2 <=1:
                sm2=SMOTE(sampling_strategy=rat2,random_state=5)
                noclass_train_sampled2,class_train_sampled2  =sm2.fit_sample(noclass_train2, class_train2.values.ravel())
                noclass_train_sampled2= pd.DataFrame(data=noclass_train_sampled2,columns =noclass_train1_cols )
                class_train_sampled2= pd.DataFrame(data=class_train_sampled2,columns =['Class'] )
                data_train3=pd.concat([noclass_train_sampled2, class_train_sampled2], axis=1)
                data_train3.loc[data_train3['machine_status']==3, 'Class'] = 1
                noclass_train3 = data_train3.loc[:, data_train3.columns != 'Class']
                class_train3 = data_train3.loc[:, data_train3.columns == 'Class']
                num_class_train3=collections.Counter(class_train3['Class'])
                numofmaj3=num_class_train3[0]
                numofmin3=num_class_train3[1]
                
                for i3 in C:
                    print("---------------")
                    print("ratio",i1)
                    results['ratio'][a1]=i3
                    
                    print("---------------")
                    print("ratio2",i2)
                    
                    print("---------------")
                    print("ratio3",i3)
                    results['ratio'][a1+1]=i2
                    results['ratio'][a1+2]=i1
                    results['Class'][a1]='NORMAL'
                    results['Class'][a1+1]='RECOVERING'
                    results['Class'][a1+2]='BROKEN'
                    
                    
                    results['Datasize'][a1]=datasize['NORMAL']
                    results['Datasize'][a1+1]=datasize['RECOVERING']
                    results['Datasize'][a1+2]=datasize['BROKEN']
                    
                    rat3=numofmin3/(numofmaj3*i3)
                    if rat3 <=1:
                        
                        i1str=str(i1)
                        i2str=str(i2)
                        i3str=str(i3)
                        rus1=RandomUnderSampler(sampling_strategy=rat3,random_state=5,replacement=True)
                        noclass_train_sampled3,class_train_sampled3  =rus1.fit_sample(noclass_train3, class_train3.values.ravel())
                        noclass_train_sampled3= pd.DataFrame(data=noclass_train_sampled3,columns =noclass_train1_cols )
                        class_train_sampled3= pd.DataFrame(data=class_train_sampled3,columns =['Class'] )
                        data_train4=pd.concat([noclass_train_sampled3, class_train_sampled3], axis=1)
                        
                        mapping3 = {1 :'NORMAL', 2 :'RECOVERING',3: 'BROKEN'}
                        data_train4=data_train4.replace({'machine_status': mapping3})
                        data_train4= data_train4.drop(['Class'],axis=1)
                        data_train4.to_csv(r'sensor_SMOTE_ratio_broken_'+i1str+'_ratio_rec_'+i2str+'_RUS_'+i3str+'_.csv')
                        mapping4 = {'NORMAL':0 ,'RECOVERING':1 ,'BROKEN':2}
                        data_train4=data_train4.replace({'machine_status': mapping4})
                        
                        
                        ratio1=str(i1)
                        ratio2=str(i2)
                        #data_train3.to_csv(r'sensor_SMOTE_ratio_broken_'+ratio1+'_ratio_rec_'+ratio2+'.csv')
                        X = data_train4.loc[:, data_train4.columns != 'machine_status']
                        y = data_train4.loc[:, data_train4.columns == 'machine_status']
        
                        #split
                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)
                        
                        print("datasize",datasize)
                        sampleddatasize=collections.Counter(data_train4['machine_status'])
                        print("sampled data size",sampleddatasize)
                        results['After Sampling'][a1]=sampleddatasize[0]
                        results['After Sampling'][a1+1]=sampleddatasize[1]
                        results['After Sampling'][a1+2]=sampleddatasize[2]
                        
                        testdatasize=collections.Counter(y_test['machine_status'])
        
                        print("testing data size",testdatasize)
                        results['Testing Datasize'][a1]=testdatasize[0]
                        results['Testing Datasize'][a1+1]=testdatasize[1]
                        results['Testing Datasize'][a1+2]=testdatasize[2]
                        
                        traindatasize=collections.Counter(y_train['machine_status'])
        
                        print("training data size",traindatasize)
                        results['Training Datasize'][a1]=traindatasize[0]
                        results['Training Datasize'][a1+1]=traindatasize[1]
                        results['Training Datasize'][a1+2]=traindatasize[2]
                        
                        dtrain = xgboost.DMatrix(data=X_train, label=y_train)
                        dtest = xgboost.DMatrix(data=X_test)
                        params = {
                                'max_depth': 3,
                                'objective': 'multi:softmax',  # error evaluation for multiclass training
                                'num_class': 3,
                                'n_gpus': 0
                                }   
                        clf= xgboost.train(params, dtrain)
                        #clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0,oob_score=True)
                        y_pred=clf.predict(dtest)
                        y_test_arr=np.array(y_test['machine_status'])
        #                oobscore=clf.oob_score_
        #                print("oob score",oobscore)
        #                results['oobscore'][a1]=round(oobscore,5)
                        #feature_imp=grid_best_clf.feature_importances_
                        #print("feature importances",feature_imp)
                        y_act_count=collections.Counter(y_test['machine_status'])
                        y_pred_count=collections.Counter(y_pred)
                        print("predicted",y_pred_count)
                        print("actual",y_act_count)
                        results['Predicted Datasize'][a1]=y_pred_count[0]
                        results['Predicted Datasize'][a1+1]=y_pred_count[1]
                        results['Predicted Datasize'][a1+2]=y_pred_count[2]
                        y_test_arr=np.array(y_test['machine_status'])
                        cn_mat=confusion_matrix( y_pred,y_test['machine_status'])
                        print(cn_mat)
                        
                        TPred_nor=cn_mat[1][1]
                        TPred_broken=cn_mat[0][0]
                        TPred_rec=cn_mat[2][2]
                        pre_nor=TPred_nor/y_pred_count[0]
                        recall_nor=TPred_nor/y_act_count[0]
                        
                        pre_broken=TPred_broken/y_pred_count[1]
                        recall_broken=TPred_broken/y_act_count[1]
                        
                        pre_rec=TPred_rec/y_pred_count[2]
                        recall_rec=TPred_rec/y_act_count[2]
                        
                        
                        results['Precision'][a1]=round(pre_nor,5)
                        results['Recall'][a1]=round(recall_nor,5)
                        
                        results['Precision'][a1+1]=round(pre_rec,5)
                        results['Recall'][a1+1]=round(recall_rec,5)
                        
                        results['Precision'][a1+2]=round(pre_broken,5)
                        results['Recall'][a1+2]=round(recall_broken,5)
                        clfres=classification_report(y_test_arr, y_pred)
                        results['Result'][a1]=clfres
                        
                        F1_nor=2*(pre_nor*recall_nor)/(pre_nor+recall_nor)
                        F1_broken=2*(pre_broken*recall_broken)/(pre_broken+recall_broken)
                        
                        F1_rec=2*(pre_rec*recall_rec)/(pre_rec+recall_rec)
                        
                        
                        results['f1-score'][a1]=round(F1_nor,5)
                        results['f1-score'][a1+1]=round(F1_rec,5)
                        
                        results['f1-score'][a1+2]=round(F1_broken,5)
                        
                        a1=a1+3
                        print("a",a1)
                        print(clfres)

        
        
        
        
