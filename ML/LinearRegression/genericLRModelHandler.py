#    Author: Shailesh Kumar
#    Email: shailesh.kmr@gmail.com
#    Date: 15/03/2021

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import metrics

def genericLinearRegressionModelHandler(function, **kwargs):
    X=kwargs['X']
    y=kwargs['y']
    
    if (kwargs['degree'] > 1):
        pol = PolynomialFeatures (degree = kwargs['degree'])
        Xpol=pol.fit_transform(X)
        pol.get_feature_names(X.columns)
        X=pd.DataFrame(Xpol, columns=pol.get_feature_names(X.columns))   
            
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.3, random_state=0 )

    if (function==LinearRegression):    
        lm=LinearRegression()
        lm.fit(X_train, y_train)
    elif (function==Ridge):
        lm=Ridge(alpha=kwargs['alpha'], normalize=kwargs['normalize'])
        lm.fit(X_train, y_train)
    elif (function==Lasso):
        lm=Lasso(alpha=kwargs['alpha'], normalize=kwargs['normalize'])
        lm.fit(X_train, y_train)
    elif (function==ElasticNet):
        lm=ElasticNet(alpha=kwargs['alpha'], normalize=kwargs['normalize'], l1_ratio=kwargs['l1_ratio'])
        lm.fit(X_train, y_train)
        
    predict=lm.predict(X_test)

    coef_dict={}
    for i in range(len(X.columns)):
        coef_dict[X.columns[i]]=lm.coef_[i]
        
    rmse=0; model_score=0
    rmse=np.sqrt(metrics.mean_squared_error(y_test, predict))
    model_score=lm.score(X_test, y_test)
         
    model_result={"intercept": lm.intercept_, "coefficient": coef_dict, "rmse": rmse, "model_score": model_score}
    
    return predict, X_train, X_test, y_train, y_test, model_result

def printModelResult(predict, y, model_result):
    print(model_result)
    coef=pd.DataFrame.from_dict(model_result['coefficient'], orient='index', columns=['Coefficients'])
    coef.sort_values(by=['Coefficients'], inplace=True)
    fig, axs = plt.subplots(ncols=2, figsize=(18,5))
    fig.tight_layout(pad=6.0)
    plt.sca(ax=axs[0]); plt.xticks(rotation=45); 
    sns.barplot(coef.index, coef['Coefficients'], ax=axs[0])
    plt.sca(ax=axs[1]); plt.xticks(rotation=45); 
    sns.distplot((y-predict),bins=50, ax=axs[1]);
