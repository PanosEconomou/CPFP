##############################################
#                                            #
#           LINEAR ALGEBRA METHODS           #
#                                            #
# This package was developed for the purpose #
# of creating a buch of useful Linear Algeb- #
# ra tools to solve some weird wavefunctions #
#                                            #
# By Panos Oikonomou (po524)                 #
#                                            #
##############################################

import numpy as np 
import scipy.constants as c
import scipy.linalg
from tqdm import tqdm

def LUdecomp(A:np.array):
    """
    This method performs LU decomposition on an NxN matrix 
    using Crout's Algortihm.

    Input:
    A: NxN numpy array

    Returns:
    L,U: NxN numpy arrays with the lower and upper factorisation
    """

    # Extract the length of the array
    N = len(A)
    
    # Create matrices
    L = np.eye(N,N)
    U = np.zeros((N,N))

    for j in tqdm(range(0,N)):
        for i in range(0,j+1):
            sum = 0
            for k in range(0,i): sum+=L[i][k]*U[k][j]
            
            U[i][j] = A[i][j] - sum
        
        for i in range(j+1,N):
            sum = 0
            for k in range(0,j): sum+=L[i][k]*U[k][j]

            if U[j][j]*(A[i][j] - sum) != 0:
                L[i][j] = 1/U[j][j]*(A[i][j] - sum)
            else: 
                L[i][j] = 0
        
    return L,U


def solveLower(L,b):
    """
    Solves the equation Lx = b using backwards decomposition

    Input:
    L:np.array. Lower triangular NxN matrix.
    b:np.array. Column vector of weights

    Returns:
    x:np.array. Solution column vector
    """
    
    N = len(b)

    if b.shape == (N,1): b = b.reshape(len(b))
    x = np.zeros(N)

    for i in tqdm(range(len(x))):
        sum = 0
        for k in range(0,i): sum += L[i][k]*x[k]

        if L[i][i]*(b[i]-sum) != 0:
            x[i] = 1/L[i][i]*(b[i]-sum)
        else:
            x[i] = 0

    return x.reshape((len(x),1))

def solveUpper(U,b):
    """
    Solves the equation Ux = b using backwards decomposition

    Input:
    U:np.array. Upper triangular NxN matrix.
    b:np.array. Column vector of weights

    Returns:
    x:np.array. Solution column vector
    """

    N = len(b)

    if b.shape == (N,1): b = b.reshape(len(b))
    x = np.zeros(N)

    for i in tqdm(range(len(x)-1,-1,-1)):
        sum = 0
        for k in range(i,N): sum += U[i][k]*x[k]

        if U[i][i]*(b[i]-sum) != 0:
            x[i] = 1/U[i][i]*(b[i]-sum)
        else:
            x[i] = 0

    return x.reshape((len(x), 1))

def solve(A,b,VERBOSE = True):
    """
    Solves the equation Ax = b using backwards LU decomposition

    Input:
    A:np.array. NxN matrix.
    b:np.array. Column vector of weights

    Returns:
    x:np.array. Solution column vector
    """

    if VERBOSE: print("LU Decomposition of Matrix A")
    # L,U = LUdecomp(A)
    L, U = scipy.linalg.lu(A, permute_l=True)
    print(L,U)
    if VERBOSE: print("LU Decomposition of Matrix A Successfull")

    if VERBOSE: print("Solving for Ly = b")
    y = solveLower(L,b)
    print(y)
    if VERBOSE: print("Solving for Ly = b Successfull")

    if VERBOSE: print("Solving for Ux = y")
    x = solveUpper(U,y)
    print(x)
    if VERBOSE: print("Solving for Ux = y Successfull")

    return x
    
def tridiag(A = None,y = None,VERBOSE = False, a = None, b = None, c = None):
    """
    Solves the equation Ax = y using Crank nicholson's Algorithm
    A has to be tridiagonal.

    Input:
    A:np.array. Tridiagonal NxN matrix.
    y:np.array. Column vector of weights

    Alternatively, instead of A, the diagonals can be entered:
    a:np.array. Array from a[2] ... a[n]
    b:np.array. Array from a[1] ... a[n]
    c:np.array. Array from a[1] ... a[n-1]

    Returns:
    x:np.array. Solution column vector
    """

    # Some houskeeping, input validation, variable declaration

    N = len(y)

    if type(A) != None:
        a = [A[i][i-1] for i in range(1,len(A))]
        b = [A[i][i] for i in range(0, len(A))]
        c = [A[i][i+1] for i in range(0, len(A)-1)]

    if y.shape == (N,1): y = y.reshape(len(y))

    cnew = np.zeros(len(c))
    ynew = np.zeros(len(y))

    # solve for cnew
    cnew[0] = c[0]/b[0]
    for i in range(1,N-1):
        cnew[i] = c[i]/(b[i]-a[i-1]*cnew[i-1])

    # solve for ynew
    ynew[0] = y[0]/b[0]
    for i in range(1,N):
        ynew[i] = (y[i]-a[i-1]*ynew[i-1])/(b[i]-a[i-1]*cnew[i-1])

    # solve for x
    x = np.zeros(N)
    x[-1] = ynew[-1]
    for i in range(N-2,-1,-1):
        x[i] = ynew[i]-cnew[i]*x[i+1]

    #Return x as a column vector
    return x.reshape((len(x), 1))
