# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# %%
import pandas as pd
import numpy as np

# %%
# while using virtual env
#df=pd.read_csv("news_article_classification_data.csv")

#using colab kernel
df=pd.read_csv("drive/MyDrive/Dataset/news_article_classification_data.csv")
df.head()
# df1=pd.read_csv("news_multilabel_merged_cleaned.csv")
# df.head()

# %%
df['labels'].unique()

# %%
df.info()

# %%
df.describe()  

# %%
print(df.duplicated().sum()) 
print(df['text'].duplicated().sum()) 
print(df['title'].duplicated().sum())


# %% [markdown]
# ## 1. Data Preprocessing & Setup
# We will parse the labels and create columns for our three scenarios (Title, Text, Title+Text).
# 

# %%
import ast
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
import numpy as np

# %%
# Convert string representation of list to actual Python list
df['label_cat_list'] = df['label_cat_list'].apply(ast.literal_eval)

# %%
df['label_cat_list'].dtype

# %%
df.info()

# %%
df['label_cat_list'].iloc[0]

# %%
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove special characters (keeping basic punctuation for BERT context)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:-]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# %%
# Apply cleaning to title and text
print("Cleaning 'title' column...")
df['title_cleaned'] = df['title'].apply(clean_text)


# %%
df['title_cleaned']

# %%
print("Cleaning 'text' column...")
df['text_cleaned'] = df['text'].apply(clean_text)


# %%
# Create the Title + Text combination
df['combined'] = df['title_cleaned'] + " - " + df['text_cleaned']

# %%
# Number of labels
num_labels = len(df['label_cat_list'].iloc[0])
print(f"Number of categories: {num_labels}")

# %%
df['label_cat_list'].dtype

# %% [markdown]
# ## 2. Tokenization and Dataset Configuration
# Here we define the PyTorch Dataset class that will tokenize the inputs using `DistilBertTokenizer`.
# 

# %%
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# %% [markdown]
# ## 3. DistilBERT Multi-Label Model Definition
# We add a linear layer with Sigmoid activation on top of the DistilBERT encoder.
# 

# %%
import torch.nn as nn

class DistilBertMultiLabelClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DistilBertMultiLabelClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier =nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 512),
            nn.ReLU(),
            
            nn.Dropout(0.2),
            nn.Linear(512,n_classes)
        )
         
        # Using BCEWithLogitsLoss later means we don't strictly need a sigmoid here in the model,
        # but we can apply it for inference. We'll leave the raw logits for the loss function.
        
    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the representation of the [CLS] token (pooled_output)
        pooled_output = output.last_hidden_state[:, 0]  # DistilBERT does not have pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# %% [markdown]
# ## 4. Prepare Data for a Single Scenario
# Here we define which text column to use, take a sample (optional), and split the data.

# %%
# For demonstration, we will use a sample and train on 'title_cleaned'.
# Change text_column to 'text_cleaned' or 'combined' to test other scenarios.
#sample_df = df.sample(n=df., random_state=42)
text_column = 'combined'

X_train, X_test, y_train, y_test = train_test_split(
    df[text_column], 
    df['label_cat_list'], 
    test_size=0.2, 
    random_state=42
)
print(f"Data Split Complete. Training set size: {len(X_train)}")

# %% [markdown]
# ## 5. Setup DataLoaders

# %%
batch_size = 32

train_dataset = NewsDataset(X_train.to_numpy(), y_train.to_numpy(), tokenizer)
test_dataset = NewsDataset(X_test.to_numpy(), y_test.to_numpy(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("DataLoaders are ready.")

# %% [markdown]
# ## 6. Model Initialization

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%

model = DistilBertMultiLabelClassifier(num_labels).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# %% [markdown]
# ## 7. Training Function

# %%
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, criterion, optimizer, device, epochs=3, val_loader=None):
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Avg Train Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f}")

        if val_loader is not None:
            eval_model(model, val_loader, device)


# %% [markdown]
# ## 8. Evaluation Function

# %%
def eval_model(model, test_loader, device):
    from sklearn.metrics import f1_score, hamming_loss, accuracy_score
    print("\nEvaluating Model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    h_loss = hamming_loss(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Accuracy:       {acc:.4f}")
    print(f"Micro F1-score: {micro_f1:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Hamming Loss:   {h_loss:.4f}")
    
    return {'accuracy': acc, 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'hamming_loss': h_loss}


# %% [markdown]
# ## 9. Execute Training and Evaluation

# %%
# Train the model with validation
train_model(model, train_loader, criterion, optimizer, device, epochs=3, val_loader=test_loader)

# Final Evaluation
results = eval_model(model, test_loader, device)


# %%


# %% [markdown]
# ## 10. Save the Trained Model
# Here we save both the custom PyTorch weights and the DistilBERT tokenizer so we can use them later without re-training.

# %%
import os

# 1. Save the Tokenizer
tokenizer.save_pretrained('saved_distilbert_tokenizer')

# 2. Save the Custom PyTorch Model Weights
torch.save(model.state_dict(), 'distilbert_multilabel_weights.pth')

print("Model and tokenizer successfully saved to disk!")

# %% [markdown]
# ## 11. Load the Model & Make Predictions
# This block demonstrates how to load your saved model from scratch and run it on a brand new news article.

# %%
# --- 1. LOAD THE SAVED ASSETS ---
print("Loading saved tokenizer and model...")
loaded_tokenizer = DistilBertTokenizer.from_pretrained('saved_distilbert_tokenizer')

# We must re-initialize the blank architecture first
loaded_model = DistilBertMultiLabelClassifier(num_labels=num_labels)

# Load the trained weights into the architecture
loaded_model.load_state_dict(torch.load('distilbert_multilabel_weights.pth', map_location=device))
loaded_model.to(device)
loaded_model.eval() # Set to evaluation mode

# %%
# --- 2. PREDICTION FUNCTION ---
def predict_news_category(raw_text, model, tokenizer, device, threshold=0.5):
    # IMPORTANT: Update this list to match the exact order of your Categories
    category_names = [
        'Politics', 'Technology', 'Business', 'Economy', 
        'International_Affairs', 'Education', 'Health', 'Science/Other'
    ]
    
    cleaned_text = clean_text(raw_text)
    
    encoding = tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        predictions_array = (probs > threshold).int().cpu().numpy()[0]
        raw_confidence = probs.cpu().numpy()[0]
    
    predicted_labels = []
    for i, is_predicted in enumerate(predictions_array):
        if is_predicted == 1:
            cat_name = category_names[i] if i < len(category_names) else f"Category_{i}"
            predicted_labels.append(cat_name)
                
    return predicted_labels, raw_confidence

# --- 3. TEST IT ---
custom_article = "The US Government passed a new law heavily regulating artificial intelligence and stock market trading."
labels, scores = predict_news_category(custom_article, loaded_model, loaded_tokenizer, device)

print(f"\nArticle: '{custom_article}'")
print(f"Predicted Categories: {labels}")
print(f"Raw Confidence Output: {np.round(scores, 3)}")

# %% [markdown]
# ## Hyperparameter Tuning with Optuna
# This cell samples a fraction of the dataset to rapidly test different transformer architectures, learning rates, and combinations without training a full model each time. Once it finds the best settings, you can use those parameters for the final dataset training.

# %%
! pip install optuna

# %%
import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
import pandas as pd

# %%
len(df)*(15/100)

# %%
# 1. Sample 15% of data to keep trials fast
sample_df = df.sample(frac=0.15, random_state=42)
# Use the combined text or title
text_column = 'combined'

# %%
X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
    sample_df[text_column], 
    sample_df['label_cat_list'], 
    test_size=0.2, 
    random_state=42
)

# %%
def objective(trial):
    # Let Optuna suggest parameters
    model_name = trial.suggest_categorical("model_name", ["distilbert-base-uncased", "roberta-base"])
    lr = trial.suggest_float("lr", 1e-5, 8e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    max_length = trial.suggest_categorical("max_length", [128, 256])
    
    epochs = 3  
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)
    
    # Calculate and print trainiable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStarting Trial {trial.number} | Model: {model_name} | Params: {num_params:,} | LR: {lr:.2e} | Batch: {batch_size} | MaxLen: {max_length}")
    
    # Save parameter count as user attribute
    trial.set_user_attr("num_params", num_params)
    
    tune_train_dataset = NewsDataset(X_tune_train.to_numpy(), y_tune_train.to_numpy(), tokenizer, max_len=max_length)
    tune_val_dataset = NewsDataset(X_tune_val.to_numpy(), y_tune_val.to_numpy(), tokenizer, max_len=max_length)
    
    tune_train_loader = DataLoader(tune_train_dataset, batch_size=batch_size, shuffle=True)
    tune_val_loader = DataLoader(tune_val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = len(tune_train_loader) * epochs
    num_warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    # Tune Training Loop
    for epoch in range(epochs):
        model.train()
        for batch in tune_train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    # Tune Validation Loop
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tune_val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate all metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    h_loss = hamming_loss(all_labels, all_preds)
    
    # Save extra metrics to Optuna's trial memory
    trial.set_user_attr("accuracy", acc)
    trial.set_user_attr("micro_f1", micro_f1)
    trial.set_user_attr("hamming_loss", h_loss)
    
    print(f"--- Trial {trial.number} Complete ---")
    print(f"Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Acc: {acc:.4f} | Hamming: {h_loss:.4f}\n")
    
    return macro_f1

# %%
study = optuna.create_study(
    study_name="transformer_tuning",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    direction="maximize"
)

# %%
study.optimize(objective, n_trials=5)

print("\n" + "="*50)
print("             OPTIMIZATION SUMMARY            ")
print("="*50)
print("Best Macro F1 Score:", study.best_trial.value)
print("Best Trial Metrics:")
print(f"  Accuracy:      {study.best_trial.user_attrs.get('accuracy', 0):.4f}")
print(f"  Micro F1:      {study.best_trial.user_attrs.get('micro_f1', 0):.4f}")
print(f"  Hamming Loss:  {study.best_trial.user_attrs.get('hamming_loss', 0):.4f}")
print("\nBest Trial Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

print("\n" + "="*50)
print("             ALL TRIALS HISTORY            ")
print("="*50)


# %%
# Fetch history as dataframe
df_trials = study.trials_dataframe()
cols_to_show = ['number', 'state', 'value', 'params_model_name', 'params_lr', 'user_attrs_accuracy', 'user_attrs_micro_f1']
available_cols = [c for c in cols_to_show if c in df_trials.columns]
if len(available_cols) > 0:
    print(df_trials[available_cols].to_string(index=False))


# %%
# uncomment below to execute
study = optuna.create_study(
    study_name="transformer_tuning",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    direction="maximize"
)
print("Starting Optuna hyperparameter sweep...")
study.optimize(objective, n_trials=10) # 5 trials to start with

print("\n--- BEST PARAMETERS FOUND ---")
print("Best Macro F1 Score:", study.best_trial.value)
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")



