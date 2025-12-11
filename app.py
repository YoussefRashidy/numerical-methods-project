from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from Controller.SolverBackend import SolverBackend
from Model.RootFinder import RootFinder
import numpy as np
from decimal import Decimal

import time

app = FastAPI()
solver = SolverBackend()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(x) for x in obj]
    return obj

class MatrixRequest(BaseModel):
    method: str
    A: List[List[float]]
    b: List[float]
    tol: Optional[float] = 0.0001
    max_iter: Optional[int] = 50
    x_init: Optional[List[float]] = None # New Parameter
    sig_figs: Optional[int] = 4          # New Parameter
    scaling: bool

@app.post("/matrix")
async def solve_matrix(data: MatrixRequest):
    try:
        x, L, U, steps, time_ms = solver.solve(
            data.method, 
            data.A, 
            data.b,
            data.scaling,
            tol=data.tol, 
            max_iter=data.max_iter,
            sig_figs=data.sig_figs,
            x_init=data.x_init
        )
        
        return make_jsonable({
            "status": "success",
            "x": x,
            "L": L, 
            "U": U,
            "steps": steps,
            "execution_time_ms": time_ms
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

class RootRequest(BaseModel):
    method: str
    equation: str
    tol: Optional[float] =1e-5
    max_iter: Optional[int] = 50
    sig_figs: Optional[int] = 5

    xl: Optional[float] = None
    xu: Optional[float] = None
    x0: Optional[float] = None
    x1: Optional[float] = None

@app.post("/root")
async def find_root(data: RootRequest):
    start = time.perf_counter()

    try:
        root, ea, iter_count, steps = RootFinder.solve(
            method = data.method,
            equation = data.equation,
            tol = data.tol,
            max_iter = data.max_iter,
            sig_figs = data.sig_figs,
            xl = data.xl, xu = data.xu,
            x0 = data.x0, x1 = data.x1,
        )

        end = time.perf_counter() # In ms
        exec_time = (end - start) * 100

        return {
            "status": "success",
            "root": root if root is not None else "Not Found",
            "ea": ea if ea is not None else 0,
            "iter": iter_count if iter_count is not None else 0,
            "steps": steps,
            "time": f"{exec_time:.4f}"
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    print("🚀 Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)