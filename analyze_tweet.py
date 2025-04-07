import gradio as gr
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch.nn as nn

##############################################################################
# Load Sentiment Model
##############################################################################
SENTIMENT_MODEL_PATH = "sentiment_model"
print(f"Loading sentiment model from: {SENTIMENT_MODEL_PATH}")

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, use_fast=False)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
# We use the label mapping: 0=negative, 1=neutral, 2=positive
sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text: str) -> str:
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=1).item()
    return sentiment_labels[pred_id]

##############################################################################
# Load Lexicon Severity Model
##############################################################################
MULTI_EMOTION_MODEL_PATH = "multi_emotion_model"
print(f"Loading multi-emotion model from: {MULTI_EMOTION_MODEL_PATH}")

class MultiOutputRegressor(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(hidden_size, 4)  # 4 outputs: anger/fear/joy/sadness

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        logits = self.regressor(x)  # shape: (batch, 4)
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            return (loss, logits)
        return (logits,)

# Load the base config to see which checkpoint was used originally
base_config = MULTI_EMOTION_MODEL_PATH  
multi_emotion_model = MultiOutputRegressor(base_config)
# Load saved weights
state_dict = torch.load(f"{MULTI_EMOTION_MODEL_PATH}/pytorch_model.bin", map_location="cpu")
multi_emotion_model.load_state_dict(state_dict)
multi_emotion_tokenizer = AutoTokenizer.from_pretrained(base_config, use_fast=False)

def predict_emotions(text: str):
    """
    Returns a tuple/list of 4 floats: [anger, fear, joy, sadness].
    """
    inputs = multi_emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = multi_emotion_model(**inputs)
    logits = outputs[0]  # shape: (1, 4)
    return logits.squeeze().tolist()  # [anger, fear, joy, sadness]


##############################################################################
# ENGAGEMENT SCORE FUNCTION
##############################################################################
# Naive Approach
def engagement_score(likes: float, retweets: float, replies: float = 0.0) -> float:
    raw_engagement = likes + retweets + replies
    return float(np.log1p(raw_engagement))  # log(1 + x)

##############################################################################
# Severity Score
##############################################################################
def final_severity(anger, fear, joy, sadness, engage, sent):
    severity = ((anger + fear + sadness + joy) / 4.0) * engage * sent 

    return severity

##############################################################################
# GRADIO INTERFACE
##############################################################################
def analyze_tweet_pipeline(text, likes, retweets, replies):
    sent = predict_sentiment(text)
    anger, fear, joy, sadness = predict_emotions(text)

    engage = engagement_score(likes, retweets, replies)
    severity_val = final_severity(anger, fear, joy, sadness, engage, sent)
    result = f"""
    
Tweet Text: {text}
Sentiment (Model #1): {sent}

Emotion Intensities (Model #2):
  - Anger:   {anger:.3f}
  - Fear:    {fear:.3f}
  - Joy:     {joy:.3f}
  - Sadness: {sadness:.3f}

Engagement Score: {engage:.3f}

FINAL SEVERITY: {severity_val:.3f}
""".strip()
    return result

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Tweet Severity Demo\nThis interface uses two models + an engagement score.")
    with gr.Row():
        with gr.Column():
            tweet_input = gr.Textbox(label="Tweet Text", lines=3, value="I'm so angry right now, I hate everything!")
            likes_input = gr.Number(label="Likes", value=50)
            retweets_input = gr.Number(label="Retweets", value=5)
            replies_input = gr.Number(label="Replies", value=2)

            submit_button = gr.Button("Analyze Tweet")
        with gr.Column():
            output_text = gr.Textbox(label="Analysis Result", lines=15)

    submit_button.click(
        fn=analyze_tweet_pipeline,
        inputs=[tweet_input, likes_input, retweets_input, replies_input],
        outputs=output_text
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()