def _fmt(val, sig_figs=4):
        """Formats a number to the specified significant figures."""
        if val == 0: return "0"
        try:
            return f"{val:.{sig_figs}g}"
        except:
            return str(val)

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