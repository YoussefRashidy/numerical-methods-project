import numpy as np
import copy
# from Model.MatrixSolver import MatrixSolver
from ..utils.matrix_utils import _log_matrix, _fmt_vec
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
      
      steps.append(f"<div class='p-2 my-2 bg-slate-800/50 border-l-2 border-yellow-500 rounded text-sm'>Pivoting: Swapped Row {k} with Row {p}</div>")
      
      for j in range(k,n) :
        dummy = A[p][j]
        A[p][j] = A[k][j]
        A[k][j] = dummy
      if b is not None:
        dummy = b[p]
        b[p] = b[k]
        b[k] = dummy
        steps.append(_log_matrix(A, b))
      else:
        #steps.append(copy.deepcopy(A))
        pass
        
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
      steps.append(f"<div class='p-2 my-2 bg-slate-800/50 border-l-2 border-yellow-500 rounded text-sm'>Pivoting: Swapped Row {k} with Row {p}</div>")
      for j in range(k,n) :
        dummy = A[p][j]
        A[p][j] = A[k][j]
        A[k][j] = dummy
      if b is not None:
        dummy = b[p]
        b[p] = b[k]
        b[k] = dummy
        steps.append(_log_matrix(A, b))
      else:
        #steps.append(copy.deepcopy(A))
        pass


def forward_elimination(A, b, s, n, tol, steps):
# A: Coef. of matrix A; 2-D array
# b: Coef. of vector b; 1-D array it is None if i didn't pass one
# n: Dimension of the system of equations
# tol: Tolerance; smallest possible scaled pivot allowed
# s: n-element array for storing scaling factors

  if s is not None and b is not None:
    for k  in range(n-1):
      Pivoting(A, b, s, n, k, steps) # Partial Pivoting
      if abs(A[k][k] / s[k]) < tol:  # Check for singularity
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        steps.append({
                        "type": "calc_l", "i": i, "j": k,
                        "formula": f"R_{{{i}}} \\leftarrow R_{{{i}}} - ({_fmt(factor)}) \\times R_{{{k}}}",
                        "res": "Row Updated"
                    })
        for j in range(k + 1,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        A[i][k] = 0  # Explicitly set to zero
        b[i] = b[i] - factor * b[k]
        steps.append(_log_matrix(A, b))

    if abs(A[n-1][n-1]/s[n-1]) < tol: # Check for singularity
      return -1
  elif s is None and b is not None: # no scaling
    for k  in range(n-1):
      Pivoting(A, b, None, n, k, steps) # Partial Pivoting
      if abs(A[k][k]) < tol:
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        steps.append({
                        "type": "calc_l", "i": i, "j": k,
                        "formula": f"R_{{{i}}} \\leftarrow R_{{{i}}} - ({_fmt(factor)}) \\times R_{{{k}}}",
                        "res": "Row Updated"
                    })
        for j in range(k + 1,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        A[i][k] = 0  # Explicitly set to zero
        b[i] = b[i] - factor * b[k]
        steps.append(_log_matrix(A, b))

    if abs(A[n-1][n-1]) < tol: # Check for singularity
      return -1
  elif s is not None and b is None:
    for k  in range(n-1):
      Pivoting(A, None, s, n, k, steps) # Partial Pivoting
      if abs(A[k][k] / s[k]) < tol:  # Check for singularity
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        steps.append({
                        "type": "calc_l", "i": i, "j": k,
                        "formula": f"R_{{{i}}} \\leftarrow R_{{{i}}} - ({_fmt(factor)}) \\times R_{{{k}}}",
                        "res": "Row Updated"
                    })
        for j in range(k + 1,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        A[i][k] = 0  # Explicitly set to zero
        steps.append(_log_matrix(A, b))

    if abs(A[n-1][n-1]/s[n-1]) < tol: # Check for singularity
      return -1
  else: # no scaling, b is None
    for k  in range(n-1):
      Pivoting(A, None, None, n, k, steps) # Partial Pivoting
      if abs(A[k][k]) < tol:
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        if abs(factor) < tol :
            continue
        steps.append({
                        "type": "calc_l", "i": i, "j": k,
                        "formula": f"R_{{{i}}} \\leftarrow R_{{{i}}} - ({_fmt(factor)}) \\times R_{{{k}}}",
                        "res": "Row Updated"
                    })
        for j in range(k + 1,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        A[i][k] = 0  # Explicitly set to zero
        steps.append(_log_matrix(A, b))

    if abs(A[n-1][n-1]) < tol: # Check for singularity
      return -1
  return 0 # Return 0 for successful elimination


def back_substitution(A, b, n):
  x = [0.0 for _ in range(n)]
  x[n-1] = b[n-1] / A[n-1][n-1]
  for i in range(n-2,-1,-1):
    sum_val = 0
    for j in range(i+1,n):
      sum_val = sum_val + A[i][j] * x[j]
    x[i] = (b[i] - sum_val) / A[i][i]
  return x

def forward_substitution(A, b, n):
    x = [0.0 for _ in range(n)]
    x[0] = b[0] / A[0][0]
    for i in range(1, n):
        sum_val = 0
        for j in range(i):
            sum_val = sum_val + A[i][j] * x[j]
        x[i] = (b[i] - sum_val) / A[i][i]
    return x

def gauss_elimination(A, b , n, tol=1e-12, scaling = False, significantFigs = 7 , rounding = True):
  """
    Method to do gauss elimination
    Inputs:
      A: Coef. of matrix A; 2-D array
      b: Coef. of vector b; 1-D array it is None if i didn't pass one
      n: Dimension of the system of equations
      tol: Tolerance; smallest possible scaled pivot allowed
      scaling: A boolean showing we want scaling or not
    Outputs:
      (x, A, steps)
  """

  intializeContext(significantFigs, rounding)
  A = toDecimal(A)
  b = toDecimal(b)

  steps = [] # initializing the steps list
  A = copy.deepcopy(A)
  b = copy.deepcopy(b) if b is not None else None
      
  steps.append("<div class='text-slate-400 mb-2'>Starting Gaussian Elimination</div>")
      
      
  if scaling:
    s = [] # as an n-element array for storing scaling factors
    for i in range(n):
      s.append(abs(A[i][0]))
      for j in range(1,n) :
        if abs(A[i][j]) > s[i]:
          s[i] = abs(A[i][j])

    if b is not None:
      temp = forward_elimination(A, b, s, n, tol, steps)  # forward elimination
      if temp != -1:             # If not singular
        return (back_substitution(A, b, n), A  , steps)   # back substitution
      else:
        raise ValueError("Singular Matrix detected during elimination.") # Indicate singular matrix
    else: # b is None
      temp = forward_elimination(A, None, s, n, tol, steps)  # forward elimination
      if temp != -1:
        return (None, A , steps) # Return the modified A if b is None
      else:
        raise ValueError("Singular Matrix detected during elimination.") # Indicate singular matrix
  else: # no scalling
      if b is not None:
        temp = forward_elimination(A, b, None, n, tol, steps)  # forward elimination, passing None for s
        if temp != -1:             # If not singular
          return (back_substitution(A, b, n), A, steps)    # back substitution
        else:
          raise ValueError("Singular Matrix detected during elimination.") # Indicate singular matrix
      else: # b is None
        temp = forward_elimination(A, None, None, n, tol, steps)  # forward elimination
        if temp != -1:
          return (None, A, steps) # Return the modified A if b is None
        else:
          raise ValueError("Singular Matrix detected during elimination.") # Indicate singular matrix

  #   temp = forward_elimination(A, b, s, n, tol, steps)  # forward elimination
    
  # else: # no scalling
  #   temp = forward_elimination(A, b, None, n, tol, steps)  # forward elimination, passing None for s
      
  #   # If forward elimination returned singular pivot
  #   if temp == -1:
  #       return None
    
  #   rankA = 0
  #   rankAb = 0

  #   for i in range(n):
  #       if not np.allclose(A[i], 0, atol=1e-12):
  #           rankA += 1
  #       if b is not None:
  #           if not np.allclose(np.append(A[i], b[i]), 0, atol=1e-12):
  #               rankAb += 1

  #   # no solution
  #   if b is not None and rankA < rankAb:
  #       return None

  #   # infinite number of solutions
  #   if rankA < n:
  #       return 2

  #   # Unique solution
  #   if b is not None:
  #       return (back_substitution(A, b, n), steps)
  #   else:
  #       return (A, steps)


def _fmt(val, sig_figs=4):
        """Formats a number to the specified significant figures."""
        if val == 0: return "0"
        try:
            return f"{val:.{sig_figs}g}"
        except:
            return str(val)