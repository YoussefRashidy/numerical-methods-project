from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from Controller.SolverBackend import SolverBackend
import numpy as np
from decimal import Decimal

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

@app.post("/solve")
async def solve_matrix(data: MatrixRequest):
    try:
        x, L, U, steps, time_ms = solver.solve(
            data.method, 
            data.A, 
            data.b, 
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

if __name__ == "__main__":
    print("🚀 Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)