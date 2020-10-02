from tkinter import *
import tkinter.filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from functions import *
import pandas as pd


class Selection_win:
    def __init__(self,master):
        #self.file_uploader = tkinter.filedialog.askopenfile(parent=root,mode='rb',title='Choose a file')
        #self.Chose_a_data = Button(master, text="Upload data",command = print("keke"))
        self.classification = Button(master,text="classification",command = self.classif_win,bg="orange")
        self.Regression = Button(master,text="Regression",command = self.regression_win,bg="orange")
        self.Dim_red = Button(master,text = "Dim. Reduction",bg="orange")
        self.geometry = "320x120"
        #self.Chose_a_data.pack()
        self.classification.pack()
        self.Regression.pack()
        self.Dim_red.pack()
        #self.Chose_a_data['command'] = self.File_open()
    def regression_win(self):
        self.newWindow = Toplevel()
        self.app = Regression_menu(self.newWindow)
        self.newWindow.geometry("350x350")
        self.newWindow.configure(bg="black")
    def classif_win(self):
        self.newWindow = Toplevel()
        self.app = Classification_menu(self.newWindow)
        self.newWindow.geometry("350x350")
        self.newWindow.configure(bg="black")

class Classification_menu:
    def __init__(self,master):
        self.Select_Features = Button(master, text="Select Features", command=self.SelectFeatures, bg="cyan")
        self.Select_labels = Button(master, text="Select labels", command=self.SelectLabels, bg="cyan")
        self.Select_labels.pack()
        self.Select_Features.pack()
        self.Decision_tree_clas = Button(master,text = "Decision tree classifier",bg="orange")
        self.Decision_tree_clas.pack()







    def SelectLabels(self):
        self.labels = tkinter.filedialog.askopenfile()
        self.labels = pd.read_csv(self.labels)
        print(self.labels)
    def SelectFeatures(self):
        self.features = tkinter.filedialog.askopenfile()
        self.features = pd.read_csv(self.features,delimiter=";")
        print(self.features)


class Regression_menu:
    def __init__(self,master):
        self.Linear_regr = Button(master, text='Linear Regression',command=self.Lin_regr,bg="orange")
        self.Select_Features = Button(master, text="Select Features",command=self.SelectFeatures,bg="cyan")
        self.Select_labels = Button(master, text="Select Targets",command=self.SelectLabels,bg = "cyan")
        self.Select_labels.pack()
        self.Select_Features.pack()
        self.Linear_regr.pack()
        self.SVR = Button(master,text="SVR",command=self.SVR_reg,bg="orange")
        self.SVR.pack()
        self.Ridge_regression = Button(master,text="Ridge",command=self.Ridge_regr,bg="orange")
        self.Lasso_regression = Button(master,text="Lasso",command=self.Lasso_regr,bg="orange")
        self.Elastic_Net = Button(master,text="Elastic Net",command=self.Elastic_net_regr,bg="orange")
        self.Lars = Button(master,text="Least angle regression",command=self.Lars_regr,bg="orange")
        self.Decision_tree_regr = Button(master,text="Decision Tree Regressor",command=self.Des_tree_regr,bg="orange")
        self.Huber = Button(master, text="Huber Regressor", command=self.Huber_regr,bg="orange")

        self.Decision_tree_regr.pack()
        self.Lars.pack()
        self.Elastic_Net.pack()
        self.Lasso_regression.pack()
        self.Ridge_regression.pack()
        self.Huber.pack()


    def Lars_regr(self):
        Lar_regr(self.features,self.labels)

    def Des_tree_regr(self):
        Des_tree_regr(self.features, self.labels)


    def Huber_regr(self):
        Huber_regressor(self.features,self.labels)


    def Elastic_net_regr(self):
        Elastic_net(self.features,self.labels)

    def SVR_reg(self):
        SVM_regr(self.features,self.labels)

    def SelectLabels(self):
        self.labels = tkinter.filedialog.askopenfile()
        self.labels = pd.read_csv(self.labels)
        print(self.labels)
    def SelectFeatures(self):
        self.features = tkinter.filedialog.askopenfile()
        self.features = pd.read_csv(self.features,delimiter=";")
        print(self.features)
    def Test(self):
        print(self.features)
        print(self.labels)

    def Lin_regr(self):
        lin_regr(self.features,self.labels)


    def Ridge_regr(self):
        Ridge_lin_regr(self.features,self.labels)

    def Lasso_regr(self):
        Lasso_Lin_regr(self.features,self.labels)

root = Tk()
selection_window = Selection_win(root)
root.geometry(selection_window.geometry)
root.configure(bg="black")
root.mainloop()