import os
import pandas as pd

def transformat():
    test_id = get_predict()
    pd_all = pd.read_csv(os.path.join("./output/test_results.csv"),sep='\t',header=None)
    data = pd.DataFrame(columns=['label'])
    print(pd_all.shape)

    for index in pd_all.index:
        negative_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]
        middle_score = pd_all.loc[index].values[2]
        if max(middle_score, positive_score, negative_score) == negative_score:
            data.loc[index ] = ["0"]
        elif max(middle_score, positive_score, negative_score) == positive_score:
            data.loc[index ] = ["1"]
        else:
            data.loc[index ] = ["2"]
    submit = pd.concat([test_id,data],axis=1)
    submit.to_csv(os.path.join("./result/submit1.csv"),index=False)

def get_predict():
    test_id = []
    with open('./data/Test/Test_DataSet.csv', encoding='utf8') as f:
        for idx, i in enumerate(f):
            if idx == 0:
                pass
            else:
                ik = str(i).split(',')[0]
                test_id.append(ik)
    test = pd.DataFrame(test_id,columns=['id'])
    # test['id'] = test_id
    return test


if __name__ == '__main__':
    transformat()

