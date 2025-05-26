# Fake News Detection API using FastAPI and BERT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline, RobertaPreTrainedModel, RobertaModel
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn

class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.relu(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

app = FastAPI()
    
#model_path = "./roberta-base-clf"
model_path = "Bluelajide/roberta-news-classifier"

# Load model and tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = CustomRobertaForSequenceClassification.from_pretrained(model_path)
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
