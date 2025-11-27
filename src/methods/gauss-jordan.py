import numpy as np
from src.utils.precision_rounding import toDecimal, toFloats, intializeContext


def Pivoting(A, b, s, n, k):
    """Partial pivoting for Decimal precision"""
    if s is not None:
        # Find pivot with largest scaled value
        pivot = k
        for i in range(k+1, n):
            if abs(A[i][k] / s[i]) > abs(A[pivot][k] / s[pivot]):
                pivot = i
    else:
        # Find pivot with largest absolute value
        pivot = k
        for i in range(k+1, n):
            if abs(A[i][k]) > abs(A[pivot][k]):
                pivot = i
    
    # Swap rows
    if pivot != k:
        A[k], A[pivot] = A[pivot], A[k]
        if b is not None:
            b[k], b[pivot] = b[pivot], b[k]
        if s is not None:
            s[k], s[pivot] = s[pivot], s[k]

def forward_elimination(A, b, s, n, tol):
  """
  A: Coef. of matrix A; 2-D array
  b: Coef. of vector b; 1-D array it is None if i didn't pass one
  n: Dimension of the system of equations
  tol: Tolerance; smallest possible scaled pivot allowed
  s: n-element array for storing scaling factors
  """

  if s is not None and b is not None:
    for k  in range(n-1):
      Pivoting(A, b, s, n, k) # Partial Pivoting
      if abs(A[k][k] / s[k]) < tol:  # Check for singularity
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        for j in range(k,n): # Changed range from k+1 to k
          A[i][j] = A[i][j] - factor * A[k][j]
        b[i] = b[i] - factor * b[k]

    if abs(A[n-1][n-1]/s[n-1]) < tol: # Check for singularity
      return -1
  elif s is None and b is not None: # no scaling
    for k  in range(n-1):
      Pivoting(A, b, None, n, k) # Partial Pivoting
      if abs(A[k][k]) < tol:
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        for j in range(k,n): # Changed range from k+1 to k
          A[i][j] = A[i][j] - factor * A[k][j]
        b[i] = b[i] - factor * b[k]

    if abs(A[n-1][n-1]) < tol: # Check for singularity
      return -1
  elif s is not None and b is None:
    for k  in range(n-1):
      Pivoting(A, None, s, n, k) # Partial Pivoting
      if abs(A[k][k] / s[k]) < tol:  # Check for singularity
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        for j in range(k,n): # Changed range from k+1 to k
          A[i][j] = A[i][j] - factor * A[k][j]

    if abs(A[n-1][n-1]/s[n-1]) < tol: # Check for singularity
      return -1
  else: # no scaling, b is None
    for k  in range(n-1):
      Pivoting(A, None, None, n, k) # Partial Pivoting
      if abs(A[k][k]) < tol:
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        for j in range(k,n): # Changed range from k+1 to k
          A[i][j] = A[i][j] - factor * A[k][j]

    if abs(A[n-1][n-1]) < tol: # Check for singularity
      return -1
  return 0 # Return 0 for successful elimination
     

def backward_elimination(A, b, s, n, tol):
    """  
    Parameters:
    A: Coefficient matrix (upper triangular after forward elimination); 2-D array
    b: Right-hand side vector; 1-D array (can be None)
    s: Scaling factors array; 1-D array (can be None)
    n: Dimension of the system
    tol: Tolerance for singularity check
    """
    
    if s is not None and b is not None:
        for k in range(n-1, 0, -1):  # Start from last row, go up to row 1
            if abs(A[k][k] / s[k]) < tol:  # Check for singularity
                return -1
            for i in range(k-1, -1, -1):  # Eliminate above pivot
                factor = A[i][k] / A[k][k]
                A[i][k] = 0.0  # Will become zero
                b[i] = b[i] - factor * b[k]
        
        # Normalize diagonal to 1 and scale solution
        for i in range(n):
            if abs(A[i][i] / s[i]) < tol:
                return -1
            b[i] = b[i] / A[i][i]
            A[i][i] = 1.0
            
    elif s is None and b is not None:  # No scaling
        for k in range(n-1, 0, -1):
            if abs(A[k][k]) < tol:  # Check for singularity
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                A[i][k] = 0.0
                b[i] = b[i] - factor * b[k]
        
        for i in range(n):
            if abs(A[i][i]) < tol:
                return -1
            b[i] = b[i] / A[i][i]
            A[i][i] = 1.0
            
    elif s is not None and b is None:  # With scaling, no b vector
        for k in range(n-1, 0, -1):
            if abs(A[k][k] / s[k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                A[i][k] = 0.0
        
        for i in range(n):
            if abs(A[i][i] / s[i]) < tol:
                return -1
            A[i][i] = 1.0
            
    else:  # No scaling, no b vector
        for k in range(n-1, 0, -1):
            if abs(A[k][k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                A[i][k] = 0.0
        
        for i in range(n):
            if abs(A[i][i]) < tol:
                return -1
            A[i][i] = 1.0
    
    return 0  # Success


def gauss_jordan(A, b, scaling, n, tol):
    """
    Parameters:
    A: Coefficient matrix; 2-D array
    b: Right-hand side vector; 1-D array (can be None)
    scaling: is scalling used or not
    n: Dimension of the system
    tol: Tolerance for singularity check
    """

    A = toDecimal(A)
    b = toDecimal(b)

    if scaling:
        s = [] # as an n-element array for storing scaling factors
        for i in range(n):
            s.append(abs(A[i][0]))
            for j in range(1,n) :
                if abs(A[i][j]) > s[i]:
                    s[i] = abs(A[i][j])

    
    # Forward elimination
    result = forward_elimination(A, b, s, n, tol)
    if result == -1:
        return None  # Singular or ill-conditioned matrix
    
    # Backward elimination
    result = backward_elimination(A, b, s, n, tol)
    if result == -1:
        return None
    
    return b  # Solution is now in b vector

