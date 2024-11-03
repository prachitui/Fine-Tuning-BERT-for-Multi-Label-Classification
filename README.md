# Fine Tuning Bert for Multi Label Classification
This project demonstrates how to fine-tune a BERT model (and similar models, such as RoBERTa, DeBERTa, etc.) for multi-label text classification—meaning that each input (in this case,
a tweet) can be assigned one or more labels from a set of possible categories. Here’s an overview of the key steps and components involved:

## Environment Setup:
The notebbok installs the necessary packages: HuggingFace's transformers and datasets libraries, which provide access to pre-trained models and various datasets.

## Dataset Loading and Preparation:
A multi-label text classification dataset is loaded from the HuggingFace Hub, specifically, the "sem_eval_2018_task_1" dataset, which contains tweets labeled with emotions. 
The dataset is split into training, validation, and test sets. The label structure is set up by extracting all the target labels (e.g., emotions) and mapping each to a unique ID for 
model use.

## Data Preprocessing: 
The notebook tokenizes text data using the BERT tokenizer (AutoTokenizer) to prepare it for input into the BERT model. Additionally, it converts labels into a matrix format where each 
tweet is represented by a vector indicating whether each label is present (1) or absent (0). This matrix is required for multi-label classification and is formatted as a floating-point
tensor.

## Model Definition: 
A bert-base-uncased model is loaded with a classification head on top. This head, a linear layer, is configured to output scores for each label. The model is set for multi-label c
lassification using BCEWithLogitsLoss (Binary Cross-Entropy with Logits Loss), which is appropriate for multi-label problems as it considers each label as a separate binary classification.

## Training Setup: 
Using the HuggingFace Trainer API, the training parameters (TrainingArguments) are configured. These parameters specify:

- Batch size
- Learning rate
- Evaluation strategy (evaluating after each epoch)
- Model saving strategy
- Number of training epochs
- Additional settings such as weight decay and loading the best model after training.
- Metric Calculation: A custom compute_metrics function is defined to calculate performance metrics (F1 score, ROC AUC, and accuracy) after each evaluation step. Predictions are
   made using a threshold (0.5) after applying the sigmoid function to model outputs, enabling interpretation as probabilities.

## Model Training and Evaluation: 
The training process begins with the Trainer.train() method, which fine-tunes the BERT model on the training set. After training, the model is evaluated on the validation set, 
with metrics providing insights into its performance.

## Inference: 
Finally, the notebook demonstrates how to use the trained model to predict labels for a new, unseen sentence. This involves:

- Tokenizing the input text.
- Running it through the model to get logits.
- Applying a sigmoid function and thresholding the logits to interpret them as label predictions.
