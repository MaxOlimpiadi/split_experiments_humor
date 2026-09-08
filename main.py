from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import random
from itertools import islice
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


#----------------------Global variables (bad practice, but i don`t care :( )----------------------------------------

# Datasets:
DATA_FOLDER_PATH = 'datasets'    
DATASET_FILE = 'colbert_dataset.csv' # put here your dataset path

# Splits:
SPLIT_FOLDER_PATH = 'split'
TRAIN_SLICES_FOLDER_PATH = os.path.join(SPLIT_FOLDER_PATH, 'train_slices')
TRAIN_SPLIT_FILE = 'train.csv'
VAL_SPLIT_FILE = 'val.csv'
TEST_SPLIT_FILE = 'test.csv'

# Test prediction results:
TEST_RESULTS_FOLDER = 'test'
TEST_RESULTS_FILE = 'predictions.csv'

# Transformer Model params:
MODEL_NAME = "bert-base-uncased"
SAVE_BEST_PATH = 'best_model.pth'
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 10
LEARNING_RATE = 5e-5

# SVM params:
ENCODING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' # for embeddings
KERNEL = 'rbf'
C = 1.0
GAMMA = 'scale'
MAX_FEATURES = 1000

# Experiment:
TRAIN_SPLIT_SIZES = (25, 50, 100, 200, 300, 500, 700, 900, 1100)
RANDOM_SEEDS = (7, 10, 35)
DELETE_OLD_REPORT = True # if you need to delete the old log file before running experiments
CREATE_SPLITS = True  # If you need to split the dataset into train, validation, and test sets. Otherwise, we load all three parts directly from the corresponding files.
LOG_FILE_NAME = "experiments_log.csv"
#------------------------------------------------------------------------------





class HumorDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item



def load_data(path):
    data = pd.read_csv(path)
    print(data["humor"].dtype)
    print(type(data["humor"].iloc[0]))
    texts = data["text"]  # Тексты шуток
    labels = data["humor"].astype(int)
    #labels = data["humor"].map({True: 1, False: 0}) # только для датасетов, где в поле humor лежать TRUE/FALSE
    
    return texts, labels


# getting a convinient structure of global train data for further slicing:
def get_samples_by_class(global_train_texts, global_train_labels, random_state):
    samples_by_class = {
        "humor": [],
        "not_humor": []
    }
    for t, l in zip(global_train_texts, global_train_labels):
        if l == 1:
            samples_by_class["humor"].append((t, l))
        elif l == 0:
            samples_by_class["not_humor"].append((t,l))  
        else: 
            raise ValueError(f"Unexpected label: {l}")
    
    # shuffle, taking care not to alter the global random state:
    rng = random.Random(random_state)
    rng.shuffle(samples_by_class["humor"])
    rng.shuffle(samples_by_class["not_humor"])
    
    return samples_by_class


def create_splits(texts, labels):
    global_train_texts, temp_texts, global_train_labels, temp_labels = train_test_split(
            texts, 
            labels,
            test_size = 0.2,
            shuffle = True,
            random_state = 42,
            stratify = labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, 
        temp_labels,
        test_size = 0.5,
        shuffle = True,
        random_state = 42,
        stratify = temp_labels
    )
    
    for seed in RANDOM_SEEDS:
        seed_folder_path = os.path.join(TRAIN_SLICES_FOLDER_PATH, f'Seed_{seed}')
        samples_by_class = get_samples_by_class(global_train_texts, global_train_labels, seed)
        for size in TRAIN_SPLIT_SIZES:
            train_texts, train_labels = create_training_slice(samples_by_class, size, seed)
            save_split(train_texts, train_labels, seed_folder_path, f'train_{size}.csv')
        
    save_split(val_texts, val_labels, SPLIT_FOLDER_PATH, VAL_SPLIT_FILE)
    save_split(test_texts, test_labels, SPLIT_FOLDER_PATH, TEST_SPLIT_FILE)



def create_training_slice(samples_by_class, size, random_state):
    n_positive = size // 2
    n_negative = size - n_positive

    # Ensure there are enough samples of each class for the requested slice size:
    if len(samples_by_class["humor"]) < n_positive:
        raise ValueError(
            f"Not enough humor samples: "
            f"requested {n_positive}, "
            f"available {len(samples_by_class['humor'])}"
        )
    if len(samples_by_class["not_humor"]) < n_negative:
        raise ValueError(
            f"Not enough non-humor samples: "
            f"requested {n_negative}, "
            f"available {len(samples_by_class['not_humor'])}"
        )
    
    positive_slice_tuples = samples_by_class["humor"][:n_positive]
    negative_slice_tuples = samples_by_class["not_humor"][:n_negative]
    
    current_data = positive_slice_tuples + negative_slice_tuples

    rng = random.Random(random_state)
    rng.shuffle(current_data)
    
    split_texts = []
    split_labels = []
    for t, l in current_data:
        split_texts.append(t)
        split_labels.append(l)
            
    return split_texts, split_labels
    
    
    
def save_split(texts, labels, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True) # if the folder doesn`t exist, it will be created`
    df = pd.DataFrame(
        {
            'text': texts,
            'label': labels
        }    
    )
    full_path = os.path.join(folder_path, filename)
    df.to_csv(full_path, index = False)
    


def load_split(file_path):
    df = pd.read_csv(file_path)  
    texts = df['text'] 
    labels = df['label']
    return texts, labels
        
    

def prepare_data(tokenizer, train_slice_file_path):
    
    train_texts, train_labels = load_split(train_slice_file_path)
    val_texts, val_labels = load_split(os.path.join(SPLIT_FOLDER_PATH, VAL_SPLIT_FILE))
    test_texts, test_labels = load_split(os.path.join(SPLIT_FOLDER_PATH, TEST_SPLIT_FILE))
    
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length = MAX_LENGTH)
    
    # Convert labels to tensors
    train_labels = torch.tensor(train_labels.values)
    val_labels = torch.tensor(val_labels.values)
    test_labels = torch.tensor(test_labels.values)

    train_dataset = HumorDataset(train_encodings, train_labels)
    val_dataset = HumorDataset(val_encodings, val_labels)
    test_dataset = HumorDataset(test_encodings, test_labels)
        
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    
    return train_dataloader, val_dataloader, test_dataloader


def create_model(model_name = MODEL_NAME, lr = LEARNING_RATE):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    optimizer = AdamW(model.parameters(), lr = lr)
    
    return model, optimizer 


def train_model(model, optimizer, train_loader, val_dataloader, device):
    # Early stopping parameters
    early_stopping_patience = 3  # Number of epochs without improvement before stopping
    best_loss = float("inf")  # Best loss value to date
    patience_counter = 0  # counter of epochs without improvement
    
    for epoch in range(EPOCHS):  # We set the maximum number of epochs.
        model.train() # Right here, because during validation, we switch the model to eval mode. So, we need to switch it back to train mode.
        print(f"Epoch {epoch + 1}")
        epoch_loss = 0
    
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {key: val.to(device) for key, val in batch.items()}
    
            # run the data through the model
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
            # Sum the Loss up!
            epoch_loss += loss.item()
    
            # Periodic Loss output
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
    
        # Average TRAIN LOSS per epoch
        avg_train_epoch_loss = epoch_loss / len(train_loader)
        print(f"Average train loss for epoch {epoch + 1}: {avg_train_epoch_loss}")
        
        # Average VAL LOSS per epoch
        avg_val_epoch_loss = validate_model(model, val_dataloader, device)
        print(f"Average val loss for epoch {epoch + 1}: {avg_val_epoch_loss}")
        
        # Stop if the average validation loss does not improve for `early_stopping_patience` epochs:
        if avg_val_epoch_loss < best_loss:
            best_loss = avg_val_epoch_loss
            patience_counter = 0
            save_model(SAVE_BEST_PATH, model, optimizer)
        else:
            patience_counter += 1 
            if patience_counter >= early_stopping_patience:
                print(f'Stopping after {early_stopping_patience} epochs without val loss improvement')
                break
    



def validate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {
                key: val.to(device)
                for key, val in batch.items()
            }
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_dataloader)

    return avg_loss



def evaluate_model(model, test_loader, tokenizer, device):
    output_file_path = os.path.join(TEST_RESULTS_FOLDER, TEST_RESULTS_FILE)  # File name for saving results
    
    # Switching the model to evaluation mode
    model.eval()
    
    results = []

    gold_labels = []
    predicted_labels = []
    
    # Prediction
    with torch.no_grad():
        for batch in test_loader:
            # Preparing a batch
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
    
            # Going throug the model
            outputs = model(**inputs)
            logits = outputs.logits
    
            # Logits to probabilities
            probabilities = softmax(logits, dim=-1)
    
            # Predicted labels
            predictions = torch.argmax(logits, dim=-1)
            
            gold_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
    
            # Saving the results
            for i in range(len(labels)):
                text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
                real_label = labels[i].item()
                predicted_label = predictions[i].item()
                prob_humorous = probabilities[i][1].item()
                prob_not_humorous = probabilities[i][0].item()
                results.append([text, real_label, predicted_label, prob_humorous, prob_not_humorous])
    
    accuracy = accuracy_score(gold_labels, predicted_labels)
    precision = precision_score(gold_labels, predicted_labels, average = 'macro')
    recall = recall_score(gold_labels, predicted_labels, average = 'macro')
    f1 = f1_score(gold_labels, predicted_labels, average = 'macro')


    # Save the results to CSV file
    results_df = pd.DataFrame(results, columns=["Text", "Real Label", "Predicted Label", "Probability Humorous", "Probability Not Humorous"])
    results_df.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Results saved to {output_file_path}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }   


def evaluate_svm_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict = True)
    return {
        "accuracy": report_dict["accuracy"],
        "precision": report_dict["macro avg"]["precision"],
        "recall": report_dict["macro avg"]["recall"],
        "f1": report_dict["macro avg"]["f1-score"]
    }




def save_model(save_path, model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),  # Weights of the model
            "optimizer_state_dict": optimizer.state_dict(),  # Optimyzer parameters
        }, 
            save_path
    )
    
    print(f"Model saved to {save_path}")




def log_experiment(
        dataset_name,
        seed_folder_name,
        train_slice, 
        size, 
        model_name, 
        lr, 
        batch_size, 
        kernel, 
        c, 
        gamma, 
        metrics_dict, 
        log_filename = LOG_FILE_NAME
):
    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "seed_folder_name": seed_folder_name,
        "train_slice": train_slice,
        "size": size,
        "model": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "kernel": kernel,
        "c": c,
        "gamma": gamma,
        **metrics_dict # unpack TP, FP, F1 end so on.
    }
    
    df = pd.DataFrame([row])
    
    # If the file does not exist, we create it with columns; if it does exist, we append to the end. (mode='a')
    header = not os.path.exists(log_filename)
    df.to_csv(log_filename, mode='a', index=False, header=header)


def plot_metrics(log_file_name, model_name_to_consider, plot_file_name):
    full_data = pd.read_csv(log_file_name)
    data = full_data[ full_data['model'] == model_name_to_consider]
    if data.empty:
        raise ValueError(f"No experiments found for model: {model_name_to_consider}")

    grouped = data.groupby('size')
    train_sizes = sorted(data["size"].unique())

    metrics = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1"
    }

    plt.figure(figsize=(10, 6))

    for metric, label in metrics.items():
        means = grouped[metric].mean().reindex(train_sizes)
        stds = grouped[metric].std().reindex(train_sizes)

        plt.plot(
            train_sizes,
            means,
            marker="o",
            linewidth=2,
            label=label
        )

        plt.fill_between(
            train_sizes,
            means - stds,
            means + stds,
            alpha=0.15
        )

    plt.title(
        "Dynamics of metrics depending on the size of the training sample"
    )
    plt.xlabel("Training sample size")
    plt.ylabel("Metric value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(
        plot_file_name,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()




def encode_data(texts, emb_model):
    X_emb = emb_model.encode(
        texts.tolist(),
        batch_size = BATCH_SIZE,
        show_progress_bar = True,
        normalize_embeddings = False  
    )
    return X_emb


def train_embedding_svm(emb_X_train, y_train):
    model = SVC(
        kernel = KERNEL, 
        C = C, 
        gamma = GAMMA
    )
    model.fit(emb_X_train, y_train)
    return model



def train_tfidf_svm(X_train, y_train):
    model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            lowercase=True,
            stop_words = "english",
            ngram_range = (1,2),
            max_features = MAX_FEATURES,
            sublinear_tf = True
        )
    ),
    (
        "method",
        SVC(
            kernel=KERNEL,   
            C=C,
            gamma=GAMMA
        )
    )
    ])
    model.fit(X_train, y_train)
    return model



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def do_transformer_experiments():
    print('----Transformer experiments----')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for seed_folder_name in os.listdir(TRAIN_SLICES_FOLDER_PATH):
        seed_folder_path = os.path.join(TRAIN_SLICES_FOLDER_PATH, seed_folder_name)
        if not os.path.isdir(seed_folder_path):
            continue
        
        print(f'Processing folder: {seed_folder_name}')
        seed = int(seed_folder_name.replace('Seed_', ''))
        for slice_file_name in os.listdir(seed_folder_path):
            set_seed(seed)
            file_path = os.path.join(seed_folder_path, slice_file_name)
            if not os.path.exists(file_path):
                print(f'The file does not exist: {file_path}')
                continue
            size = int(
                slice_file_name
                .replace('train_', '')
                .replace('.csv', '')
            )

            print(f"\n\tProcessing the training slice: {file_path}\n")
            model, optimizer = create_model() # # Important: initialize a fresh model for each train split
            model.to(device)

            train_dataset, val_dataset, test_dataset = prepare_data(tokenizer, file_path)
            train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset)
            
            train_model(model, optimizer, train_dataloader, val_dataloader, device)
            checkpoint = torch.load(SAVE_BEST_PATH)
            model.load_state_dict(checkpoint["model_state_dict"])
        
            report_dict = evaluate_model(model, test_dataloader, tokenizer, device)
            print(report_dict)
            log_experiment(
                dataset_name = DATASET_FILE,
                seed_folder_name = seed_folder_name,
                train_slice = slice_file_name, 
                size = size, 
                model_name = MODEL_NAME, 
                lr = LEARNING_RATE, 
                batch_size = BATCH_SIZE, 
                kernel = '-', 
                c = '-', 
                gamma = '-', 
                metrics_dict = report_dict, 
            )   

    plot_metrics(LOG_FILE_NAME, MODEL_NAME, 'bert_plot.png')




def do_svm_experiments(vectorizer_type, emb_model=None):
    print(f'----SVM-{vectorizer_type} experiments----')
    test_texts, test_labels = load_split(os.path.join(SPLIT_FOLDER_PATH, TEST_SPLIT_FILE))
    report_model_name = f'SVM_{vectorizer_type}'
    
    for seed_folder_name in os.listdir(TRAIN_SLICES_FOLDER_PATH):
        seed_folder_path = os.path.join(TRAIN_SLICES_FOLDER_PATH, seed_folder_name)
        if not os.path.isdir(seed_folder_path):
            continue

        print(f'Processing folder: {seed_folder_name}')
        for slice_file_name in os.listdir(seed_folder_path):
            file_path = os.path.join(seed_folder_path, slice_file_name)
            if not os.path.exists(file_path):
                print(f'The file does not exist: {file_path}')
                continue
            size = int(
                slice_file_name
                .replace('train_', '')
                .replace('.csv', '')
            )

            print(f"\n\tProcessing the training slice: {file_path}\n")
            train_texts, train_labels = load_split(file_path)
            if vectorizer_type == 'embeddings':
                emb_texts_train = encode_data(train_texts, emb_model)
                emb_texts_test = encode_data(test_texts, emb_model)
                model = train_embedding_svm(emb_texts_train, train_labels)
                report_dict = evaluate_svm_model(emb_texts_test, test_labels, model)
                print(report_dict)

            elif vectorizer_type == 'tf-idf':
                model = train_tfidf_svm(train_texts, train_labels)
                report_dict = evaluate_svm_model(test_texts, test_labels, model)
                print(report_dict)
            else:
                raise ValueError(
                    f"Unknown vectorizer: {vectorizer_type}. "
                    "Permissible values : 'tf-idf', 'embeddings'."
                )

            log_experiment(
                dataset_name = DATASET_FILE,
                seed_folder_name = seed_folder_name,
                train_slice = slice_file_name, 
                size = size, 
                model_name = report_model_name, 
                lr = '-', 
                batch_size = '-', 
                kernel = KERNEL, 
                c = C, 
                gamma = GAMMA, 
                metrics_dict = report_dict, 
            )    

    plot_metrics(LOG_FILE_NAME, report_model_name, f'SVM_{vectorizer_type}_plot.png')



def main():
    if DELETE_OLD_REPORT and os.path.exists(LOG_FILE_NAME): # delete the entire log before a new series of experiments
        os.remove(LOG_FILE_NAME) 
    if CREATE_SPLITS:
        data_path = os.path.join(DATA_FOLDER_PATH, DATASET_FILE)
        texts, labels = load_data(data_path)
        create_splits(texts, labels)

    embedding_model = SentenceTransformer(ENCODING_MODEL)
    do_svm_experiments('embeddings', embedding_model)
    do_svm_experiments('tf-idf')
    do_transformer_experiments()
    
    

if __name__ == "__main__":
    main()
    
    
    