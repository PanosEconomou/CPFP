####################################################
#       This is a Burlish-Stoer Integrator         #
#                                                  #
# This script can be used for any application in   #
# our computational class.                         #
#                                                  #
#  Created by Panos Oikonomou (po524)              #
####################################################


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def mag(x: np.array):
    return (x.dot(x))**0.5

def ModifiedMidMethod(grad,x_0:float,y_0:float,dx:float,n:int = 4,VERBOSE:bool = False):

    """
    This runs a modified midpoint method to calculate the next step point at x_0 + dx with 
    gradient grad(x,y)
    
    Inputs: -------------------------------
    Required:
    grad: 2 Input fucntion, returns one float. The gradient function for the system i.e. grad(x,y) = h
    x_0 : float. Start point x coordinate
    y_0 : float. Start point y coordinate
    dx  : float. Step Length
    
    Optional
    n   : int. Number of equipartitioned points inside dx. (Default Value = 4)
    VERBOSE: bool. Flag for printing a bunch of stuff.
    """

    Y_n2 = y_0
    Y_n1 = y_0 + dx/n *grad(x_0,y_0)
    Y_n  = 0

    #Calculate y
    for i in range(2,n+1):
        Y_n  = Y_n2 + 2 * dx/n*grad(x_0+(i-1)*dx/n, Y_n1)
        Y_n2 = Y_n1
        Y_n1 = Y_n
    
    Y = 0.5*(Y_n+Y_n2+dx/n*grad(x_0+dx,Y_n))

    if VERBOSE: print(Y)
    
    return Y

def LagrangeFit(x,points):
    """
    This fits a curve to the points with the lagrange method.

    Inputs: -------------------------------
    Required:
    x     : x-coordinate for the estimate
    points: The list (or array) of the points=[[x1,y1],[x2,y2]...[xn,yn]]
    """

    sum = 0

    for i in range(len(points[0])):

        l = 1
        for n in range(len(points[0])):
            if n == i: continue
            l *= (x-points[0][n])/(points[0][i]-points[0][n])

        sum+=l*(points[1][i])
        
    return sum


def BurlishStoer(x_0:float,y_0:float,grad,dx:float,mindiff=10e-10, VERBOSE:bool = False):
    """
    This performs one step as a Burlish Stoer integrator

    Inputs: -------------------------------
    Required:
    grad: 2 Input fucntion, returns one float. The gradient function for the system i.e. grad(x,y) = h
    x_0 : float. Start point x coordinate
    y_0 : float. Start point y coordinate
    dx  : float. Step Length
    
    Optional
    diff   : float. This is the difference that should make it stop. (Default Value = 10e-15)
    """

    n = 4
    pts = []
    for i in range(0,2): 
        n*=2
        pts.append([dx/n,ModifiedMidMethod(grad,x_0,y_0,dx,n=n)])
    
    x_now  = LagrangeFit(0,np.array(pts).T)
    x_prev = x_now*10
    check = [max(np.abs(x)) for x in x_now-x_prev]

    while max(check) > mindiff:
        n*=2
        x_prev = x_now
        pts.append([dx/n,ModifiedMidMethod(grad, x_0, y_0, dx, n=n)])
        x_now = LagrangeFit(0,np.array(pts).T)
        check = [max(np.abs(x)) for x in x_now-x_prev]

        if VERBOSE:
            print(n)
            print("x_now:",x_now)
            print("x_prev:", x_prev)
            print("x_now-x_prev:", x_now-x_prev)
            print("check:", check)
            print("pts: ",pts)

    return x_now


