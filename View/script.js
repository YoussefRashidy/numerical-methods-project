// --- 1. Grid Generation ---
function generateGrid() {
    const n = parseInt(document.getElementById('dimInput').value);
    const matrixDiv = document.getElementById('matrixContainer');
    const vectorDiv = document.getElementById('vectorContainer');
    const initGuessDiv = document.getElementById('initialGuessContainer');
    
    // Layout Styles
    matrixDiv.style.gridTemplateColumns = `repeat(${n}, minmax(60px, 1fr))`;
    vectorDiv.style.gridTemplateRows = `repeat(${n}, 1fr)`;
    
    // Clear old inputs
    matrixDiv.innerHTML = '';
    vectorDiv.innerHTML = '';
    initGuessDiv.innerHTML = '';

    // Matrix A Inputs
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'matrix-input w-full h-10 rounded text-sm';
            input.placeholder = (i === j) ? '1' : '0';
            input.id = `a-${i}-${j}`;
            matrixDiv.appendChild(input);
        }
    }

    // Vector b Inputs
    for (let i = 0; i < n; i++) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'matrix-input w-full h-10 rounded text-sm border-yellow-900/50 bg-yellow-900/10 focus:border-yellow-500';
        input.placeholder = '0';
        input.id = `b-${i}`;
        vectorDiv.appendChild(input);
    }

    // Initial Guess Inputs (x_init)
    for (let i = 0; i < n; i++) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'matrix-input w-20 h-10 rounded text-sm border-cyan-900/50 bg-cyan-900/10 focus:border-cyan-500';
        input.placeholder = '0';
        input.id = `xinit-${i}`;
        initGuessDiv.appendChild(input);
    }
    
    document.getElementById('resultSection').classList.add('hidden');
}

// --- 2. UI Toggling ---
function toggleParams() {
    const method = document.getElementById('methodSelect').value;
    const paramsDiv = document.getElementById('iterativeParams');
    const scalingDiv = document.getElementById('scalingParam');
    
    if (method.includes("Iteration")) {
        paramsDiv.classList.remove('hidden');
    } else {
        paramsDiv.classList.add('hidden');
    }

    if (method == "Crout Decomposition" || method == "Gauss-Jordan" || method == "Gaussian Elimination"){
        scalingDiv.classList.remove('hidden');
    }
    else{
        scalingDiv.classList.add('hidden');
    }
}

// --- 3. API Call ---
async function solveSystem() {
    const n = parseInt(document.getElementById('dimInput').value);
    const method = document.getElementById('methodSelect').value;
    const tol = parseFloat(document.getElementById('tolInput').value);
    const maxIter = parseInt(document.getElementById('iterInput').value);
    const sigFigs = parseInt(document.getElementById('sigFigsInput').value);
    const scalingInput = document.getElementById('scalingInput')
    const scaling = scalingInput ? scalingInput.checked : false;
    
    let A = [];
    let b = [];
    let x_init = [];

    try {
        // Read Matrix A and b
        for (let i = 0; i < n; i++) {
            let row = [];
            for (let j = 0; j < n; j++) {
                let val = document.getElementById(`a-${i}-${j}`).value;
                row.push(parseFloat(val || 0));
            }
            A.push(row);
            let valB = document.getElementById(`b-${i}`).value;
            b.push(parseFloat(valB || 0));
        }

        // Read Initial Guess (Only if needed, but safe to read always)
        if (method.includes("Iteration")) {
            for (let i = 0; i < n; i++) {
                let val = document.getElementById(`xinit-${i}`).value;
                x_init.push(parseFloat(val || 0));
            }
        }
    } catch (e) {
        alert("Invalid Input numbers"); return;
    }

    const btn = document.querySelector('.btn-primary');
    const originalText = btn.innerText;
    btn.innerText = "CALCULATING...";
    btn.disabled = true;

    try {
        const response = await fetch('http://localhost:8000/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                method: method,
                A: A,
                b: b,
                tol: tol,
                max_iter: maxIter,
                x_init: x_init.length > 0 ? x_init : null,
                sig_figs: sigFigs,
                scaling: scaling,
            })
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || "Server Error");

        displayResults(data.x, data.L, data.U, data.steps, data.execution_time_ms, sigFigs);

    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

// --- 4. Display Logic ---
function displayResults(x, L, U, steps, timeMs, sigFigs) {
    const section = document.getElementById('resultSection');
    section.classList.remove('hidden');
    section.classList.add('fade-in');

    // Display Time
    document.getElementById('execTime').innerText = timeMs.toFixed(2);

    // Helper to format numbers based on user's sig figs request
    // (Though backend should handle most steps, x vector usually comes raw)
    const fmt = (num) => {
        if (num === 0) return "0";
        return Number(num).toPrecision(sigFigs).replace(/\.0+$/, "");
    };

    // Render Vector X
    let xHtml = `\\[ \\begin{bmatrix} ${x.map(v => fmt(v)).join('\\\\')} \\end{bmatrix} \\]`;
    document.getElementById('solutionVector').innerHTML = xHtml;

    // Render Matrix L
    const lContainer = document.getElementById('matrixLContainer');
    if (L) {
        lContainer.classList.remove('hidden');
        let lRows = L.map(row => row.map(v => fmt(v)).join(' & ')).join(' \\\\ ');
        document.getElementById('matrixLDisplay').innerHTML = `\\[ L = \\begin{bmatrix} ${lRows} \\end{bmatrix} \\]`;
    } else {
        lContainer.classList.add('hidden');
    }

    // Render Matrix U
    const uContainer = document.getElementById('matrixUContainer');
    if (U) {
        uContainer.classList.remove('hidden');
        let uRows = U.map(row => row.map(v => fmt(v)).join(' & ')).join(' \\\\ ');
        document.getElementById('matrixUDisplay').innerHTML = `\\[ U = \\begin{bmatrix} ${uRows} \\end{bmatrix} \\]`;
    } else {
        uContainer.classList.add('hidden');
    }

    // Render Steps
    const stepsLog = document.getElementById('stepsLog');
    stepsLog.innerHTML = "";
    if (steps && steps.length > 0) {
        steps.forEach(s => {
            let div = document.createElement('div');
            div.className = "p-3 border-b border-slate-700 last:border-0";
            
            if (s.type === 'calc_l' || s.type === 'calc_u' || s.type === 'diag' || s.type === 'off') {
                div.innerHTML = `
                    <div class="text-cyan-400 text-sm font-bold mb-1">Calculating ${s.type.includes('u') ? 'U' : 'L'}(${s.i},${s.j})</div>
                    <div class="text-sm opacity-80 mb-1">\\[ ${s.formula} \\]</div>
                    <div class="flex items-center gap-3 text-sm">
                        <span class="text-emerald-400 font-bold"> = ${s.res}</span>
                    </div>
                `;
            } else if (s.type === 'iter') {
                // Formatting for iterative steps
                div.innerHTML = `
                    <div class="flex justify-between items-center text-sm">
                        <span class="text-cyan-400 font-bold">Iteration ${s.k}</span>
                        <span class="text-slate-400">Error: ${s.error}%</span>
                    </div>
                    <div class="text-emerald-300 mt-1 font-mono text-xs">
                        ${s.steps.map((step, idx) => `${step}`).join('<br>')}
                        <br>
                        x = [${s.x_vec.join(", ")}]
                    </div>
                `;
            } else {
                div.innerHTML = `<div class="text-sm text-slate-300">${s}</div>`;
            }
            stepsLog.appendChild(div);
        });
    } else {
        stepsLog.innerHTML = "<div class='text-slate-500 italic'>No detailed steps available for this method.</div>";
    }

    
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\(', right: '\\)', display: false},
            {left: '\\[', right: '\\]', display: true}
        ]
    });
    
    section.scrollIntoView({ behavior: 'smooth' });

}

function toggleSteps() {
    const container = document.getElementById('stepsContainer');
    const btn = document.getElementById('stepsBtn');
    if (container.classList.contains('hidden')) {
        container.classList.remove('hidden');
        btn.innerText = "Hide Detailed Steps ▲";
    } else {
        container.classList.add('hidden');
        btn.innerText = "Show Detailed Steps ▼";
    }
}

// Init
generateGrid();