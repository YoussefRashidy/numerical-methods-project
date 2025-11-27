import numpy as np
import copy
from src.utils.precision_rounding import toDecimal, toFloats, intializeContext


def Pivoting(A, b, s, n, k, steps):
# A: Coef. of matrix A; 2-D array
# b: Coef. of vector b; 1-D array it is None if i didn't pass one
# n: Dimension of the system of equations
# s: n-element array for storing scaling factors
# k: is the row we are in now

  # initializing p (our pivoting row) with the value of k
  p = k     
    
  if s is not None:

    # Finding the largest scaled coefficient in column k
    big = abs(A[k][k] / s[k])
    for i in range(k+1,n) :
      dummy = abs(A[i][k] / s[i]) # dummy number for the scaled value
      if dummy > big:
        big = dummy
        p = i # new pivoting row

    # Swap row p and row k if p != k
    if p != k:
      # Swaping row p and row k
      
      steps.append("A[" + str(p) + "] <-> A[" + str(k) + "]")
      
      for j in range(k,n) :
        dummy = A[p][j]
        A[p][j] = A[k][j]
        A[k][j] = dummy
      if b is not None:
        dummy = b[p]
        b[p] = b[k]
        b[k] = dummy
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))
      else:
        steps.append(copy.deepcopy(A))
        
      dummy = s[p]
      s[p] = s[k]
      s[k] = dummy
  else:
  # Finding the largest coefficient in column k
    big = abs(A[k][k])
    for i in range(k+1,n) :
      dummy = abs(A[i][k] ) # dummy number for the  value
      if dummy > big:
        big = dummy
        p = i # new pivoting row

    # Swap row p and row k if p != k
    if p != k:
      steps.append("A[" + str(p) + "] <-> A[" + str(k) + "]")
      for j in range(k,n) :
        dummy = A[p][j]
        A[p][j] = A[k][j]
        A[k][j] = dummy
      if b is not None:
        dummy = b[p]
        b[p] = b[k]
        b[k] = dummy
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))
      else:
        steps.append(copy.deepcopy(A))


def forward_elimination(A, b, s, n, tol, steps):
    """
    A: Coef. of matrix A; 2-D array
    b: Coef. of vector b; 1-D array it is None if i didn't pass one
    n: Dimension of the system of equations
    tol: Tolerance; smallest possible scaled pivot allowed
    s: n-element array for storing scaling factors
    """
    if s is not None and b is not None:
        for k in range(n-1):
            Pivoting(A, b, s, n, k, steps) # Partial Pivoting
            if abs(A[k][k] / s[k]) < tol: # Check for singularity
                return -1
            for i in range(k+1, n):
                factor = A[i][k] / A[k][k]
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                for j in range(k, n):
                    A[i][j] = A[i][j] - factor * A[k][j]
                b[i] = b[i] - factor * b[k]
                steps.append((copy.deepcopy(A), copy.deepcopy(b)))
        
        if abs(A[n-1][n-1] / s[n-1]) < tol:
            return -1
    elif s is None and b is not None: # no scaling
        for k in range(n-1):
            Pivoting(A, b, None, n, k, steps)
            if abs(A[k][k]) < tol:
                return -1
            for i in range(k+1, n):
                factor = A[i][k] / A[k][k]
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                for j in range(k, n):
                    A[i][j] = A[i][j] - factor * A[k][j]
                b[i] = b[i] - factor * b[k]
                steps.append((copy.deepcopy(A), copy.deepcopy(b)))
        
        if abs(A[n-1][n-1]) < tol:
            return -1

    elif s is not None and b is None:
        for k in range(n-1):
            Pivoting(A, None, s, n, k, steps)
            if abs(A[k][k] / s[k]) < tol:
                return -1
            for i in range(k+1, n):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                for j in range(k, n):
                    A[i][j] = A[i][j] - factor * A[k][j]
                steps.append(copy.deepcopy(A))
        
        if abs(A[n-1][n-1] / s[n-1]) < tol:
            return -1

    else:
        for k in range(n-1):
            Pivoting(A, None, None, n, k, steps)
            if abs(A[k][k]) < tol:
                return -1
            for i in range(k+1, n):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                for j in range(k, n):
                    A[i][j] = A[i][j] - factor * A[k][j]
                steps.append(copy.deepcopy(A))
        
        if abs(A[n-1][n-1]) < tol:
            return -1

    return 0

     

def backward_elimination(A, b, s, n, tol, steps):
    """  
    Parameters:
    A: Coefficient matrix (upper triangular after forward elimination); 2-D array
    b: Right-hand side vector; 1-D array (can be None)
    s: Scaling factors array; 1-D array (can be None)
    n: Dimension of the system
    tol: Tolerance for singularity check
    """
    
    if s is not None and b is not None:
        for k in range(n-1, 0, -1):
            if abs(A[k][k] / s[k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                A[i][k] = 0
                b[i] = b[i] - factor * b[k]
                steps.append((copy.deepcopy(A), copy.deepcopy(b)))
        
        # Normalize diagonal
        for i in range(n):
            if abs(A[i][i] / s[i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            b[i] = b[i] / A[i][i]
            A[i][i] = 1
            steps.append((copy.deepcopy(A), copy.deepcopy(b)))
    
    elif s is None and b is not None:
        for k in range(n-1, 0, -1):
            if abs(A[k][k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                A[i][k] = 0
                b[i] = b[i] - factor * b[k]
                steps.append((copy.deepcopy(A), copy.deepcopy(b)))
        
        for i in range(n):
            if abs(A[i][i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            b[i] = b[i] / A[i][i]
            A[i][i] = 1
            steps.append((copy.deepcopy(A), copy.deepcopy(b)))
    
    elif s is not None and b is None:
        for k in range(n-1, 0, -1):
            if abs(A[k][k] / s[k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                A[i][k] = 0
                steps.append(copy.deepcopy(A))
        
        for i in range(n):
            if abs(A[i][i] / s[i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            A[i][i] = 1
            steps.append(copy.deepcopy(A))
    
    else:
        for k in range(n-1, 0, -1):
            if abs(A[k][k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append(f"A[{i}] -> A[{i}] + {-factor} * A[{k}]")
                A[i][k] = 0
                steps.append(copy.deepcopy(A))
        
        for i in range(n):
            if abs(A[i][i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            A[i][i] = 1
            steps.append(copy.deepcopy(A))
    
    return 0


def gauss_jordan(A, b, scaling, n, tol, significantFigs = 7 , rounding = True):
    """
    Parameters:
    A: Coefficient matrix; 2-D array
    b: Right-hand side vector; 1-D array (can be None)
    scaling: is scalling used or not
    n: Dimension of the system
    tol: Tolerance for singularity check
    """
    intializeContext(significantFigs, rounding)

    A = toDecimal(A)
    b = toDecimal(b)

    if scaling:
        s = [] # as an n-element array for storing scaling factors
        for i in range(n):
            s.append(abs(A[i][0]))
            for j in range(1,n) :
                if abs(A[i][j]) > s[i]:
                    s[i] = abs(A[i][j])

    steps = []    
    # Store initial state
    if b is not None:
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))
    else:
        steps.append(copy.deepcopy(A))
    
    # Forward elimination
    result = forward_elimination(A, b, s, n, tol, steps)
    if result == -1:
        return None
    
    # Backward elimination
    result = backward_elimination(A, b, s, n, tol, steps)
    if result == -1:
        return None
    
    if b is not None:
        return (b, steps)
    else:
        return (A, steps)

