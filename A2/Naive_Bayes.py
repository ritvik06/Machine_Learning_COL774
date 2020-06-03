import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
import random
from sklearn.metrics import confusion_matrix 
import scikitplot as skplt
from matplotlib import pyplot as plt

df = np.asarray(pd.read_csv("./training.csv",encoding='latin-1'))
df_test = np.asarray(pd.read_csv("./test.csv",encoding='latin-1'))

def drop_digits(in_str):
    digit_list = "1234567890"
    for char in digit_list:
        in_str = in_str.replace(char, "")

    return in_str

#Create dictionary
#Key is the word, value is the dictionary number

dict = {}
dict_pos = {}
dict_neg = {}
# selected_keys = list()
dict_bi = {}
dict_tri = {}
dict_bipos = {}
dict_bineg = {}
dict_tripos = {}
dict_trineg = {}
count = 0
mode = 3

d = ",.!?/.&-:;% "

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = set(stopwords.words('english')) 


ps =PorterStemmer()

for i in range(len(df)):
#     print(df.iloc[i][5] + '\n')
    str = drop_digits(df[i][5])
    arr = ' '.join(w for w in re.split("["+"\\".join(d)+"]", str) if w).lower().split(' ')
    arr = [w for w in arr if not w in stop_words]
#     print(arr[0][0])
    if(len(arr)!=0):
        if arr[0][0]=='@':
            arr.pop(0)
#     tweets.append(arr)

    if(mode==1):
        for j in range(len(arr)):
#             arr[j] = ps.stem(arr[j])
            if arr[j] not in dict.keys():
                random = np.zeros((3))
                #[Count, freq_pos, freq_neg]

                random[0] = count
                count+=1      
                if i<(800000):
                    random[1]=0
                    random[2]=1
                else:
                    random[1]=1
                    random[2]=0

                dict[arr[j]] = random

            else:
                random = dict[arr[j]]
                if i<800000:    
                    random[2]+=1
                else:
                    random[1]+=1
                dict[arr[j]] = random
    elif(mode==2):
        for j in range(len(arr)-1):
#             arr[j] = ps.stem(arr[j])
            if (arr[j],arr[j+1]) not in dict_bi.keys():
                random = np.zeros((3))
                #[Count, freq_pos, freq_neg]

                random[0] = count
                count+=1      
                if i<(800000):
                    random[1]=0
                    random[2]=1
                else:
                    random[1]=1
                    random[2]=0

                dict_bi[(arr[j],arr[j+1])] = random

            else:
                random = dict_bi[(arr[j],arr[j+1])]
                if i<800000:    
                    random[2]+=1
                else:
                    random[1]+=1
                dict_bi[(arr[j],arr[j+1])] = random
        
        #Adding Mode 1 to this as well

        for j in range(len(arr)):
    #         arr[j] = ps.stem(arr[j])
            if arr[j] not in dict.keys():
                random = np.zeros((3))
                #[Count, freq_pos, freq_neg]

                random[0] = count
                count+=1      
                if i<(800000):
                    random[1]=0
                    random[2]=1
                else:
                    random[1]=1
                    random[2]=0

                dict[arr[j]] = random

            else:
                random = dict[arr[j]]
                if i<800000:    
                    random[2]+=1
                else:
                    random[1]+=1
                dict[arr[j]] = random

    elif(mode==3):
        for j in range(len(arr)-2):
#             arr[j] = ps.stem(arr[j])
            if (arr[j],arr[j+1],arr[j+2]) not in dict_tri.keys():
                random = np.zeros((3))
                #[Count, freq_pos, freq_neg]

                random[0] = count
                count+=1      
                if i<(800000):
                    random[1]=0
                    random[2]=1
                else:
                    random[1]=1
                    random[2]=0

                dict_tri[(arr[j],arr[j+1],arr[j+2])] = random

            else:
                random = dict_tri[(arr[j],arr[j+1],arr[j+2])]
                if i<800000:    
                    random[2]+=1
                else:
                    random[1]+=1
                dict_tri[(arr[j],arr[j+1],arr[j+2])] = random
        
        #Adding Mode 1 to this as well

        for j in range(len(arr)):
    #         arr[j] = ps.stem(arr[j])
            if arr[j] not in dict.keys():
                random = np.zeros((3))
                #[Count, freq_pos, freq_neg]

                random[0] = count
                count+=1      
                if i<(800000):
                    random[1]=0
                    random[2]=1
                else:
                    random[1]=1
                    random[2]=0

                dict[arr[j]] = random

            else:
                random = dict[arr[j]]
                if i<800000:    
                    random[2]+=1
                else:
                    random[1]+=1
                dict[arr[j]] = random


    
    
    
print(len(dict))
# print(dict)
# print(len(tweets))

if mode==2:  
    print(dict_bi[('never','looked')])
    count=0
    for key in dict_bi.keys():
        if (count<5):
            print(key,": ",dict_bi[key])
        count+=1

size=0
size_bi= 0
size_tri = 0
for i in range(len(df)):
    str = drop_digits(df[i][5])
    arr = ' '.join(w for w in re.split("["+"\\".join(d)+"]", str) if w).lower().split(' ')
    size+= len(arr)

size_bi = size - (len(df))

size_tri = size_bi - (len(df))
# print(size)

# if (mode==1):
for key in dict.keys():
#     Calculate parameter with laplace smoothening
    random = dict[key]
    dict_neg[key] = float((random[2]+1)/(size+len(dict)))
    dict_pos[key] = float((random[1]+1)/(size+len(dict)))
# prob_is_spam = 0.5
if (mode==2):
    for key in dict_bi.keys():
    #     Calculate parameter with laplace smoothening
        random = dict_bi[key]
        dict_bineg[key] = float((random[2]+1)/(size_bi+len(dict)))
        dict_bipos[key] = float((random[1]+1)/(size_bi+len(dict)))
elif (mode==3):
    for key in dict_tri.keys():
    #     Calculate parameter with laplace smoothening
        random = dict_tri[key]
        dict_trineg[key] = float((random[2]+1)/(size_tri+len(dict)))
        dict_tripos[key] = float((random[1]+1)/(size_tri+len(dict)))
    # print(dict_bineg[('is','upset')])
#     print(dict_bineg[('never','looked')])
#     print(dict_bipos[('never','looked')])
#     print(dict_bipos[('never','looked')]/(dict_bineg[('never','looked')] + dict_bipos[('never','looked')]))

correct,total = (0,0)
ps = PorterStemmer()
d = ",.!?/..&-:;...% "
# d = "!#$%&'()*+,-./:;?@[\]^_`...{|}~.. "
y_pred = []
y_test = []
y_probas = []
for i in range(len(df_test)):
    if df_test[i][0]!=2:
        y_test.append(df_test[i][0]/4)
        pred=0
        pos_sum,neg_sum = (1,1)
        pos_bisum,neg_bisum = (1,1)
        pos_trisum,neg_trisum = (1,1)
        str = drop_digits(df_test[i][5])
        arr = ' '.join(w for w in re.split("["+"\\".join(d)+"]", str) if w).lower().split(' ')
        arr = [w for w in arr if not w in stop_words]
        if(len(arr)!=0):
            if arr[0][0]=='@':
                arr.pop(0)
#         if (mode==1):
        for j in range(len(arr)): 
#             arr[j] = ps.stem(arr[j])
            if arr[j] in dict.keys():
                pos_sum*=dict_pos[arr[j]]
                neg_sum*=dict_neg[arr[j]]
        if (mode==2):
            for j in range(len(arr)-1): 
    #             arr[j] = ps.stem(arr[j])
                if (arr[j],arr[j+1]) in dict_bi.keys():
    #                 print("Entered")
                    pos_bisum*=dict_bipos[(arr[j],arr[j+1])]
                    neg_bisum*=dict_bineg[(arr[j],arr[j+1])]
        elif (mode==3):
            for j in range(len(arr)-2): 
    #             arr[j] = ps.stem(arr[j])
                if (arr[j],arr[j+1],arr[j+2]) in dict_tri.keys():
    #                 print("Entered")
                    pos_trisum*=dict_tripos[(arr[j],arr[j+1],arr[j+2])]
                    neg_trisum*=dict_trineg[(arr[j],arr[j+1],arr[j+2])]
#         prob = float((pos_bisum)/((pos_bisum)+(neg_bisum)))
#         prob = float((pos_sum)/(pos_sum + neg_sum))
#         prob = float((pos_bisum*pos_sum)/((pos_bisum*pos_sum)+(neg_bisum*neg_sum)))
        prob = float((pos_trisum*pos_sum)/((pos_trisum*pos_sum)+(neg_trisum*neg_sum)))
        y_probas.append(prob)

        if prob>0.5:
            pred=4 
        else:
            pred=0
        y_pred.append(pred/4)
        if(pred==df_test[i][0]): 
            correct+=1
        total+=1
#         print("answer ",pred)
print('Correct:',correct,'/',total)
print('Accuracy:',float(correct*100/total),'%')

confusion_matrix(y_test, y_pred)

#===================================RANDOM=======================================
correct,total = (0,0)

for i in range(len(df_test)):
    if df_test[i][0]!=2:
        total+=1
        n = random.randint(0,1)
        if(4*n==df_test[i][0]):
            correct+=1
print('Correct:',correct,'/',total)
print('Accuracy:',float(correct*100/total),'%')

#===================================MAX==========================================
correct,total = (0,0)

count_pos, count_neg = (0,0)

for i in range(len(df)):
    if(df[i][0]==0):
        count_neg+=1
    elif(df[i][0]==4):
        count_pos+=1

if count_neg>count_pos:
    y = 0
else:
    y = 4

for i in range(len(df_test)):
    if df_test[i][0]!=2:
        total+=1
        if(y==df_test[i][0]):
            correct+=1
print('Correct:',correct,'/',total)
print('Accuracy:',float(correct*100/total),'%')

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

#===============================PLOTTING ROC CURVE================================
# fpr,tpr,= roc_curve(y_test,y_probas)
fpr, tpr, thresholds = roc_curve(y_test, y_probas)
plot_roc_curve(fpr,tpr)
