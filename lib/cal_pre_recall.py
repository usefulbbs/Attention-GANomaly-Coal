import os.path as osp
import os
import json
import sys


def cal_pre_recall(dctT,dctF):
    tp,fp,fn=0,0,0
    for k,v in dctT:
        if k==0:
            tp+=1
        elif k==1:
            fp+=1
    for k,v in dctF:
        if k==0:
            fn+=1
    prec=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    return prec,recall,fp,fn

def cal_pre_recall1(dctT,dctF):
    tp,fp,fn=0,0,0
    for k,v in dctT:
        if k==1:
            fn+=1
        #     tp+=1
        # elif k==1:
        #     fp+=1
    for k,v in dctF:
        if k==1:
            tp+=1
        elif k==0:
            fp+=1
    prec=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    return prec,recall,fp,fn

def list2dct(score_lst, label_lst):
    label2score=[]
    for i in range(len(score_lst)):
        label2score.append((label_lst[i],score_lst[i]))
    tp_lst=sorted(label2score, key=lambda x:x[1],reverse=False)

    dct_600=tp_lst[0:600]
    dct_back=tp_lst[600:]

    pre,recall,fp,fn=cal_pre_recall1(dct_600, dct_back)
    return pre,recall,fp,fn

'''
{
    'score'=[],
    'label'=[]
}
'''
import os.path as osp
import json
def get_single_auc(json_path):
    jsonPath=osp.join(json_path)
    res=json.loads(open(jsonPath).read())
    pre,rec,fp,fn=list2dct(res['scores'], res['labels'])
    print('precison={},recall={}'.format(pre, cal_pre_recall))


def cal_all(dirpath):
    print('name'+'\t'+'precision'+'\t'+'recall')
    for filename in os.listdir(dirpath):
        if filename.endswith('json'):
            jsonPath=osp.join(dirpath,filename)
            res=json.loads(open(jsonPath).read())
            pre,rec,fp,fn=list2dct(res['scores'], res['labels'])
            print(str(filename)+'\t'+str(pre)+'\t'+str(rec))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong args')