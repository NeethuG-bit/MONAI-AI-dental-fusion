from fastapi import FastAPI
import torch

app = FastAPI()

@app.post("/infer")
def run_inference(data: dict):
    # load model
    # run inference
    return {"status": "done"}