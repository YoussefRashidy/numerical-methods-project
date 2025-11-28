import numpy as np
import copy
from src.utils.precision_rounding import toDecimal, toFloats, intializeContext

def _fmt(val, sig_figs=4):
        """Formats a number to the specified significant figures."""
        if val == 0: return "0"
        try:
            return f"{val:.{sig_figs}g}"
        except:
            return str(val)
        
@staticmethod
def _log_matrix(A, b, sig_figs=4):
    """Helper to format the current matrix state as a clean HTML block."""
    rows = len(A)
    # Container with dark background and monospaced font
    html = "<div class='mt-2 mb-4 inline-block bg-slate-900 rounded p-3 font-mono text-xs border border-slate-700 shadow-inner'>"
    
    for i in range(rows):
        # Format row values with padding for alignment
        row_str = "  ".join([f"{_fmt(val, sig_figs):>8}" for val in A[i]])
        
        # Add the augmented vector 'b' if it exists
        b_str = f"  |  {_fmt(b[i], sig_figs)}" if b is not None else ""
        
        # Combine into a single line
        html += f"<div class='whitespace-pre hover:bg-slate-800/50 px-1 rounded'>{row_str}{b_str}</div>"
    
    html += "</div>"
    return html

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
        steps.append(_log_matrix(A, b))
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
        steps.append(_log_matrix(A, b))
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
                steps.append({
                   "type": "calc_l",
                   "i": i,
                   "j": k,
                   "formula": f"R_{{i}} \\leftarrow R_{{i}} - ({_fmt(factor)}) \\times R_{{k}}",
                   "res": "Row updated"
                })
                A[i][k] = 0
                b[i] = b[i] - factor * b[k]
                steps.append(_log_matrix(A, b))
        
        # Normalize diagonal
        for i in range(n):
            if abs(A[i][i] / s[i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            b[i] = b[i] / A[i][i]
            A[i][i] = 1
            steps.append(_log_matrix(A, b))
    
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
                steps.append(_log_matrix(A, b))
        
        for i in range(n):
            if abs(A[i][i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            b[i] = b[i] / A[i][i]
            A[i][i] = 1
            steps.append(_log_matrix(A, b))
    
    elif s is not None and b is None:
        for k in range(n-1, 0, -1):
            if abs(A[k][k] / s[k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append({
                   "type": "calc_l",
                   "i": i,
                   "j": k,
                   "formula": f"R_{{i}} \\leftarrow R_{{i}} - ({_fmt(factor)}) \\times R_{{k}}",
                   "res": "Row updated"
                })
                A[i][k] = 0
                steps.append(_log_matrix(A, b))
        
        for i in range(n):
            if abs(A[i][i] / s[i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            A[i][i] = 1
            steps.append(_log_matrix(A, b))
    
    else:
        for k in range(n-1, 0, -1):
            if abs(A[k][k]) < tol:
                return -1
            for i in range(k-1, -1, -1):
                factor = A[i][k] / A[k][k]
                if abs(factor) < tol:
                    continue
                steps.append({
                   "type": "calc_l",
                   "i": i,
                   "j": k,
                   "formula": f"R_{{i}} \\leftarrow R_{{i}} - ({_fmt(factor)}) \\times R_{{k}}",
                   "res": "Row updated"
                })
                A[i][k] = 0
                steps.append(_log_matrix(A, b))
        
        for i in range(n):
            if abs(A[i][i]) < tol:
                return -1
            factor = A[i][i]
            steps.append(f"A[{i}] -> A[{i}] / {factor}")
            A[i][i] = 1
            steps.append(_log_matrix(A, b))
    
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
    s = None
    if scaling:
        s = [] # as an n-element array for storing scaling factors
        for i in range(n):
            s.append(abs(A[i][0]))
            for j in range(1,n) :
                if abs(A[i][j]) > s[i]:
                    s[i] = abs(A[i][j])

    steps = [] # initializing the steps list
    A = copy.deepcopy(A)
    b = copy.deepcopy(b) if b is not None else None
        
    steps.append("<div class='text-slate-400 mb-2'>Starting Gaussian Elimination</div>")
    
    # Forward elimination
    result = forward_elimination(A, b, s, n, tol, steps)
    if result == -1:
        raise ValueError("Singular Matrix detected during elimination.") # Indicate singular matrix
    
    # Backward elimination
    result = backward_elimination(A, b, s, n, tol, steps)
    if result == -1:
        raise ValueError("Singular Matrix detected during elimination.") # Indicate singular matrix
    
    return (b, A, steps)
