import pandas as pd
from sklearn.decomposition import PCA

def select_features(x, num_components):
    pca = PCA(n_components=num_components)
    x1 = pca.fit_transform(x)
    principal_c = pd.DataFrame(data=x1,columns=[f'Principal Component {i}' for i in range(num_components)])
    return principal_c