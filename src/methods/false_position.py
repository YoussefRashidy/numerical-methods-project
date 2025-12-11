import decimal
import pandas as pd # pandas for the table
import sympy as sp

def intializeContext(significantFigs,rounding) :
    #Intialize the context with specified signFigs and rounding/chopping
    decimal.getcontext().prec = significantFigs 
    if( rounding ) :
        decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    else :
        decimal.getcontext().rounding = decimal.ROUND_DOWN 

def parse_exp(expr_str , var_name='x') :
    x = sp.symbols(var_name)
    expr = sp.sympify(expr_str)
    
    def decimal_func(x_decimal) :
        if not isinstance(x_decimal,decimal.Decimal) :
            x_decimal = decimal.Decimal(x_decimal)
            
        result = expr.evalf(decimal.getcontext().prec,subs={x:x_decimal})
        return decimal.Decimal(str(result))
    
    return decimal_func 

def false_position(x1, x2, func_expr, tol=decimal.Decimal("1e-7"), max_iter=100,
                  significantFigs=6, rounding=True):
    
    # initialize precision
    intializeContext(significantFigs, rounding)

    # convert to Decimal
    x1 = decimal.Decimal(str(x1))
    x2 = decimal.Decimal(str(x2))

    # parse expression to Decimal-based function
    f = parse_exp(func_expr)
    
    if f(x1) * f(x2) > 0: #checking whether the bounds have odd number of roots or not
      print("the bounds are wrong")
      return None, None # Return None for xr and None for table

    if f(x1) * f(x2) == 0:
      if f(x1) == 0:
          return x1, None # Return xr and None for table
      else:
          return x2, None # Return xr and None for table

    xl = min(x1,x2)
    xu = max(x1,x2)

    iteration_data = [] # List to store iteration details

    xr_old = ((xl*f(xu)) - (xu*f(xl))) / (f(xu)-f(xl)) * 0.5

    for i in range(1,max_iter+1):
      xr=((xl*f(xu)) - (xu*f(xl))) / (f(xu)-f(xl))
      iteration_data.append({
        "Iteration": i,
        "xl": xl,
        "xu": xu,
        "xr": xr,
        "f(xr)": f(xr),
        "Et": abs(xr - xr_old),
        "ϵ": str(abs((xr - xr_old)/xr)) + "%"
      })

      if(f(xr) *f(xl) < 0):
        xu=xr
      else:
        xl=xr

      if i != 1 and abs(xr - xr_old) < tol:
        break

      xr_old = xr # Updating xr_old for the next iteration

    df = pd.DataFrame(iteration_data)
    return xr, df