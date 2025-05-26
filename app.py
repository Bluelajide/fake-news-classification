# Fake News Detection API using FastAPI and BERT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaModel, RobertaPreTrainedModel
import torch.nn as nn

app = FastAPI()
    
#model_path = "./roberta-base-clf"
model_path = "Bluelajide/roberta-news-classifier"

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

        if not input_text.strip():
            return {"Label": "Empty", "Confidence": 0.0}

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        id2label = model.config.id2label
        if isinstance(id2label, dict):
            label = id2label.get(str(pred_idx), id2label.get(int(pred_idx), str(pred_idx)))
        else:
            label = str(pred_idx)
        confidence = float(probs[pred_idx])

        return {
            "Label": label,
            "Confidence": confidence
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Add this line
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

