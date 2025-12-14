// --- 1. UI Logic ---
function updateUI() {
    const method = document.getElementById('methodSelect').value;
    
    const containers = {
        bracketing: document.getElementById('bracketingParams'),
        initial: document.getElementById('initialGuessParams'),
        secant: document.getElementById('secantParams'),
        gEq: document.getElementById('gEquationContainer')
    };

    // Reset all
    Object.values(containers).forEach(el => el.classList.add('hidden'));

    // Show specific
    if (method === 'Bisection' || method === 'False-Position') {
        containers.bracketing.classList.remove('hidden');
    } 
    else if (method === 'Newton-Raphson' || method === 'Modified Newton') {
        containers.initial.classList.remove('hidden');
    } 
    else if (method === 'Fixed Point') {
        containers.initial.classList.remove('hidden');
        containers.gEq.classList.remove('hidden'); // Show g(x) input
    }
    else if (method === 'Secant') {
        containers.secant.classList.remove('hidden');
    }
}

// --- 2. Plotting Logic ---
function plotFunction() {
    const method = document.getElementById('methodSelect').value;
    const fExpr = document.getElementById('equationInput').value;
    const xStart = parseFloat(document.getElementById('plotStart').value) || -10;
    const xEnd = parseFloat(document.getElementById('plotEnd').value) || 10;
    
    // JS Math Safety Helper
    const parseMath = (expr, xVal) => {
        try {
            const safe = expr
                .replace(/\^/g, '**')
                .replace(/\bsin\b/g, 'Math.sin')
                .replace(/\bcos\b/g, 'Math.cos')
                .replace(/\btan\b/g, 'Math.tan')
                .replace(/\bexp\b/g, 'Math.exp')
                .replace(/\blog\b/g, 'Math.log')
                .replace(/\bsqrt\b/g, 'Math.sqrt')
                .replace(/\be\b/g, 'Math.E');

            return eval(safe.replace(/\bx\b/g, `(${xVal})`));
        } catch (e) {
            return null;
        }
    };


    const xVals = [];
    const yVals = [];
    const y2Vals = []; // For y=x or y=0

    // Range: -10 to 10 (Could be dynamic based on guess)
    for (let x = xStart; x <= xEnd; x += 0.1) {
        xVals.push(x);
        
        if (method === 'Fixed Point') {
            // Plot g(x) and y=x
            const gExpr = document.getElementById('gEquationInput').value;
            yVals.push(parseMath(gExpr, x)); // g(x)
            y2Vals.push(x); // y=x
        } else {
            // Plot f(x) and y=0
            yVals.push(parseMath(fExpr, x)); // f(x)
            y2Vals.push(0); // x-axis
        }
    }

    // Config Traces
    const trace1 = {
        x: xVals, y: yVals,
        mode: 'lines',
        name: method === 'Fixed Point' ? 'g(x)' : 'f(x)',
        line: {color: '#c084fc', width: 3}
    };

    const trace2 = {
        x: xVals, y: y2Vals,
        mode: 'lines',
        name: method === 'Fixed Point' ? 'y = x' : 'y = 0',
        line: {color: '#94a3b8', width: 2, dash: 'dash'}
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#cbd5e1' },
        margin: { t: 20, r: 20, b: 40, l: 40 },
        xaxis: { gridcolor: '#334155', zerolinecolor: '#94a3b8' },
        yaxis: { gridcolor: '#334155', zerolinecolor: '#94a3b8' },
        showlegend: true,
        legend: { x: 0, y: 1 }
    };

    const config = {
        scrollZoom: true,       // Enable mouse wheel zoom
        responsive: true,       // Resize when window changes
        displayModeBar: true,   // Show the tool bar
        displaylogo: false,     // Hide Plotly logo for clean look
        modeBarButtonsToRemove: ['lasso2d', 'select2d'] // Remove useless buttons
    };


    Plotly.newPlot('plotDiv', [trace1, trace2], layout, config);
}


// --- 3. API Call ---
async function solveRoot() {
    // A. Gather Common Inputs
    const method = document.getElementById('methodSelect').value;
    const equation = document.getElementById('equationInput').value;
    const tol = parseFloat(document.getElementById('epsilonInput').value);
    const maxIter = parseInt(document.getElementById('maxIterInput').value);
    const sigFigs = parseInt(document.getElementById('sigFigsInput').value);
    const gEquation = document.getElementById('gEquationInput').value;

    // B. Gather Method-Specific Inputs
    let xl = null, xu = null, x0 = null, x1 = null;

    if (method === 'Bisection' || method === 'False-Position') {
        xl = parseFloat(document.getElementById('xlInput').value);
        xu = parseFloat(document.getElementById('xuInput').value);
        if (isNaN(xl) || isNaN(xu)) { 
            alert("Please enter valid brackets (Xl, Xu)"); return; 
        }
    } 
    else if (method === 'Newton-Raphson' || method === 'Modified Newton' || method === 'Fixed Point') {
        x0 = parseFloat(document.getElementById('x0Input').value);
        if (isNaN(x0)) { 
            alert("Please enter an initial guess (X0)"); return; 
        }
    } 
    else if (method === 'Secant') {
        x0 = parseFloat(document.getElementById('x_minus_1').value); // First guess
        x1 = parseFloat(document.getElementById('x_i').value);       // Second guess
        if (isNaN(x0) || isNaN(x1)) { 
            alert("Please enter both guesses for Secant"); return; 
        }
    }

    // C. UI Loading State
    const btn = document.querySelector('button[onclick="solveRoot()"]');
    const originalText = btn.innerText;
    btn.innerText = "Finding Root...";
    btn.disabled = true;

    try {
        // D. Send Request to Python Backend
        const response = await fetch('http://localhost:8000/root', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                method: method,
                equation: equation,
                tol: tol,
                max_iter: maxIter,
                sig_figs: sigFigs,
                // Pass optional params (backend handles nulls)
                xl: xl, 
                xu: xu, 
                x0: x0, 
                x1: x1,
                g_equation: gEquation
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Server Error");
        }

        // E. Update UI with Results
        displayResults(data);

    } catch (err) {
        console.log("Error: ", err.message)
        alert("Method diverged!");
    } finally {
        // F. Reset UI State
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

// --- 4. Display Results (Helper) ---
function displayResults(data) {
    const section = document.getElementById('resultSection');
    section.classList.remove('hidden');
    
    // Fill Summary Cards
    document.getElementById('rootDisplay').innerText = data.root;
    document.getElementById('errorDisplay').innerText = data.ea + "%"; // Relative Error
    document.getElementById('iterDisplay').innerText = data.iter;
    document.getElementById('execTime').innerText = data.time + "ms";

    // Fill Steps Table
    const tbody = document.getElementById('stepsTableBody');
    tbody.innerHTML = "";
    
    if (data.steps && data.steps.length > 0) {
        data.steps.forEach(step => {
            let row = `<tr class="border-b border-slate-700 hover:bg-slate-700/50">
                <td class="px-6 py-3 font-mono text-cyan-400">${step.iter}</td>
                <td class="px-6 py-3 font-mono">${step.x_old}</td>
                <td class="px-6 py-3 font-mono font-bold text-white">${step.x_new}</td>
                <td class="px-6 py-3 font-mono text-yellow-400">${step.error}%</td>
            </tr>`;
            tbody.innerHTML += row;
        });
    } else {
        tbody.innerHTML = `<tr><td colspan="4" class="text-center py-4 text-slate-500 italic">No detailed steps available</td></tr>`;
    }
    
    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth' });
}

function toggleSteps() {
    document.getElementById('stepsContainer').classList.toggle('hidden');
}

// Initialize
updateUI();

document.addEventListener("DOMContentLoaded", function(){
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\(', right: '\\)', display: false},
            {left: '\\[', right: '\\]', display: true}
        ]
    });
})