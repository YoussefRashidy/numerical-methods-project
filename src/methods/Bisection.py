import decimal
import pandas as pd #pandas for the table 
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

def Bisection(x1, x2, func_expr, tol=decimal.Decimal("1e-7"), max_iter=100,
                  significantFigs=6, rounding=True):
    
    # initialize precision
    intializeContext(significantFigs, rounding)

    # convert to Decimal
    x1 = decimal.Decimal(str(x1))
    x2 = decimal.Decimal(str(x2))

    # parse expression to Decimal-based function
    f = parse_exp(func_expr)
    f_xl = f(x1)
    f_xu = f(x2)

    if f_xl * f_xu > 0:  #checking whether the bounds have odd number of roots or not
        raise ValueError("Invalid Interval: No root bracketed (f(xl)*f(xu) > 0)")

    if  f_xl * f_xu == 0:
        if f(x1) == 0:
            return x1, None # Return xr and None for table
        else:
            return x2, None # Return xr and None for table

    xl = min(x1,x2)
    xu = max(x1,x2)

    iteration_data = [] # List to store iteration details

    # xr = (xl+xu)/decimal.Decimal("4.0")
    xr_old = None

    for i in range(1,max_iter+1):
        xr=(xl+xu)/decimal.Decimal("2.0")

        if xr_old is None:
            ea_rel = "---"
        else:
            if xr == 0:
                ea_rel = "Undefined"
            else:
                ea_rel = str(abs((xr - xr_old)/xr) * 100)

        iteration_data.append({
            "iter": i,
            "xl": str(xl),
            "xu": str(xu),
            "x_new": str(xr),
            "f(xr)": str(f(xr)),
            "x_old": str(xr_old),
            "Et": str(abs(xr - xr_old)) if xr_old is not None else "---",
            "error": ea_rel
        })

        if(f(xr) *f(xl) < 0):
            xu=xr
        else:
            xl=xr

        if i != 1 and abs(xr - xr_old) < tol: 
            break

        xr_old = xr # Update xr_old for the next iteration

    # df = pd.DataFrame(iteration_data)
    return (xr, ea_rel, len(iteration_data), iteration_data)

