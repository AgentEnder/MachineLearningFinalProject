import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer

def preprocess(data, save_encoded = False):
    le = LabelEncoder()
    one_hot = OneHotEncoder()

    data = data.apply(le.fit_transform) # Apply label encoder to each column
    # features 9,10,11,12 need to be one hot encoded.
    
    before_encoded_section = data.iloc[:,:8] #Don't run over these columns
    
    to_encode = data.iloc[:,8:12] #Encode these
    to_encode = one_hot.fit_transform(to_encode)

    to_encode = pd.DataFrame(to_encode.toarray())


    after_encoded_section = data.iloc[:,12:] #Do nothing to these
    

    data = before_encoded_section.join(to_encode) #Join the encoded data
    data = data.join(after_encoded_section) #join the remaining data

    if save_encoded:
        data.to_csv('data/encoded_mat.csv')

    return data
    
def split_attributes(data, num_labels):
    x = data.iloc[:,:-1*num_labels]
    y = data.iloc[:,num_labels*-1:]
    return x, y

def bucketize_y(y, num_buckets):
    binner = KBinsDiscretizer(n_bins = num_buckets, encode='ordinal')
    cols = y.columns
    y = binner.fit_transform(y)
    y = pd.DataFrame(data=y,columns=cols)
    return y