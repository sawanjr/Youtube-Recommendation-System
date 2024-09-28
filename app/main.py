
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from src.models.model import ConcatNet
import yaml
# Initialize FastAPI app
app = FastAPI()

with open("src/config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)


# Set up Jinja2 template renderer
templates = Jinja2Templates(directory="app/templates")


# Define request schema
class UserItemRequest(BaseModel):
    user_id: int
    item_id: int

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConcatNet({
    'num_users': config['num_users'],
    'num_items': config['num_items'],
    'emb_size': config['emb_size'],
    'emb_dropout': config['emb_dropout'],
    'fc_layer_sizes': config['fc_layer_sizes'],
    'dropout': config['dropout'],
    'out_range': config['out_range']
}).to(device)

# Load the model and its weights
model = ConcatNet(config).to(device)
model.load_state_dict(torch.load("saved_models/best_model.pth"))
model.eval()  # Set the model to evaluation mode

# Home endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict/")
async def predict_rating(request: UserItemRequest):
    # Convert user_id and item_id to tensors
    user_id_tensor = torch.tensor([request.user_id], dtype=torch.long, device=device)
    item_id_tensor = torch.tensor([request.item_id], dtype=torch.long, device=device)
    
    # Predict the rating
    with torch.no_grad():
        pred_rating = model(user_id_tensor, item_id_tensor)
    
    # Convert tensor to a scalar value
    rating = pred_rating.item()
    
    # Return the predicted rating as a JSON response
    return {"predicted_rating": rating}

# Run the app using 'uvicorn' if not executed in a Jupyter environment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
