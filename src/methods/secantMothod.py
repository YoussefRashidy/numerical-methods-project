

from src.utils.precision_rounding import intializeContext
import decimal

from src.utils.roots_utils import parse_exp

def secant_method(x0, x1, func_expr, tol=decimal.Decimal("1e-7"), max_iter=100,
                  significantFigs=14, rounding=True):

    # initialize precision
    intializeContext(significantFigs, rounding)

    # convert to Decimal
    x0 = decimal.Decimal(str(x0))
    x1 = decimal.Decimal(str(x1))

    # parse expression to Decimal-based function
    f = parse_exp(func_expr)

    # evaluate function at initial guesses
    f0 = f(x0)
    f1 = f(x1)

    iteration_details = []
    # relativeError = decimal.Decimal("Infinity")
    relativeError = "---"

    for i in range(max_iter):

        # avoid division by zero
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Secant method failed: Division by zero (f1 - f0 = 0).")

        # Secant formula
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        x_new = decimal.Decimal(str(x_new))

        # compute error (starting from iteration 2)
        if i != 0:
            relativeError = (abs(x_new - x1) / x_new) * 100

        # build evaluation string
        expr_eval_str = f"({x1} - f(x1)*(x1-x0)/(f1-f0)) = {x_new}"

        # save iteration details
        details = {
            "iter": str(i + 1),
            "x_old": str(x1),
            "x_new": str(x_new),
            "error": relativeError,
            "Evaluation": expr_eval_str
        }
        iteration_details.append(details)

        # check tolerance
        if abs(x_new - x1) <= tol:
            break

        # shift values
        x0, x1 = x1, x_new
        f0, f1 = f1, f(x_new)

    return (x_new, relativeError, len(iteration_details) ,iteration_details)


