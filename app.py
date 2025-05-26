# Fake News Detection API using FastAPI and BERT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
    
model_path = "./roberta-base-clf"
#model_path = "Bluelajide/roberta-news-classifier"

# Load model and tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load pipeline
news_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Serve static files (including index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def read_index():
    return FileResponse("index.html")

# --- Prediction endpoint ---
@app.post("/predict")
def predict(request: NewsRequest):
    try:
        input_text = request.text

        if len(input_text) > 2000:
            raise HTTPException(status_code=400, detail="Input is too long. Please limit to 2000 characters.")

        print("INPUT TO PIPELINE:", input_text)

        if not input_text.strip():
            return {"Label": "Empty", "Confidence": 0.0}

        result = news_pipe(input_text)[0]
        print("PIPELINE RESULT:", result)
        return {
            "Label": result['label'],
            "Confidence": float(result['score'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
