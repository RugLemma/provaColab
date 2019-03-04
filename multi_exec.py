
# USAGE
# py multi_exec.py
import os
import json
import numpy as np
num_dir = 0
dir_name = []
prec = []
recall = []
spec = []
acc = []


num_exec=2

for j in range(0,num_exec):
    print("Esecuzione numero "+str(j))
    os.system("py save_model.py --model saved_model.model -p "+str(j)+".png")
    os.system("py load_model.py -m saved_model.model")


for root, dirs, files in os.walk('Report'):
    if dirs!=[] :
        dir_name = dirs
        print(dirs)
        print(dir_name)

#for usato per contare le directory in Report senza contare la dir []

#print(num_dir)


# leggere valori dal json
for i in range(0,len(dir_name)):
    print(dir_name[i])
    folder_path="Report/"+dir_name[i]+"/metrics.json"
    file_j = open(folder_path,"r")
    '''
    for line in file_j:
        print(line)
    '''
    data = json.load(file_j)
    '''
    print(data["Precision"])
    prec.append(data["Precision"])
    print(prec)
    '''
    #Commenti per precision non richiesti attualmente
    acc.append(data["Accuracy"])
    spec.append(data["Specificity"])
    recall.append(data["Recall"])


'''
print(len(dir_name))
media=accumulatore/len(dir_name)
print("Precisione media "+str(media))

prec_ave = np.mean(prec)
print(prec_ave)
'''

acc_average = np.mean(acc)
spec_average = np.mean(spec)
recall_average = np.mean(recall)

print(acc)
print(spec)
print(recall)

acc_std = np.std(acc)
spec_std = np.std(spec)
recall_std = np.std(recall)

acc_str = "Accuracy mean ± standard deviation = " + str(acc_average) + " ± " + str(acc_std)
spec_str = "Specificity mean ± standard deviation = " + str(spec_average) + " ± " + str(spec_std)
recall_str = "Recall mean ± standard deviation = " + str(recall_average) + " ± " + str(recall_std)

print(acc_str)
print(spec_str)
print(recall_str)

