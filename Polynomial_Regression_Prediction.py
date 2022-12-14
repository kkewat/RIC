# Polynomial Regresion Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_coef(x,y):
    #number of observations/points
    n = np.size(x)

    #mean of x and y vector
    m_x,m_y = np.mean(x),np.mean(y)

    #calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    #calculating regression coefficients
    b_1 =SS_xy/SS_xx
    b_0 = m_y - b_1 *m_x

    return(b_0,b_1)

def plot_regression_line(x,y,b):
    #plotting the actual points as scatter plot
    plt.scatter(x,y,color="m",marker="o",s=30)

    #predicted response vector
    y_pred =b[0] + b[1] * x

    #Plotting the Regression Line
    plt.plot(x,y_pred,color="g")

    #puting Labels
    plt.xlabel('x')
    plt.ylabel('y')

    #Function to show plot
    plt.show()

def main():
    #Onservations
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1,3,2,5,7,8,8,9,10,12])

    #estimate coefficients
    b=estimate_coef(x,y)
    print("Estimated Coefficients: \n b_0 = {}".format(b[0],b[1]))

    #plotting Regression Line
    plot_regression_line(x,y,b)

if __name__ =="__main__":
    main()
