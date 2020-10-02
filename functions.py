from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def AsGraph(labels,predict):
    Axis = []
    for i in range(1, len(labels) + 1):
        Axis.append(i)
    print(Axis)
    print(labels)
    plt.plot(Axis, labels, label="Real Data")
    plt.plot(Axis, predict, label="Model Data")
    error = mean_squared_error(labels,predict)
    print(error)
    plt.Figure()
    plt.title(error)
    plt.legend()
    plt.show()

#Regressions Algorithms
def SVM_regr(features,labels):
    from sklearn.svm import SVR
    model = SVR()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)

def Ridge_lin_regr(features,labels):
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)

def Lasso_Lin_regr(features,labels):
    from sklearn.linear_model import Lasso
    model = Lasso()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)




def Elastic_net(features,labels):
    from sklearn.linear_model import ElasticNet
    model = ElasticNet()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)

def Des_tree_regr(features,labels):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)



def Lar_regr(features,labels):
    from sklearn.linear_model import Lars
    model = Lars()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)


def lin_regr(features,labels):
    model = LinearRegression()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)

def Huber_regressor(features,labels):
    from sklearn.linear_model import HuberRegressor
    model = HuberRegressor()
    model.fit(features,labels)
    pred = model.predict(features)
    AsGraph(labels,pred)




