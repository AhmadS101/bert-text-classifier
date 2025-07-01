# BERT Classifier Model 

## **Project Overview**
This project implements a multi-class text classification model using [**BERT**](https://huggingface.co/google-bert/bert-base-uncased)  as its base architecture. The model adds a **classification head**  ( neural network layer added on top of BERT that performs the final classification) on top of the pre-trained BERT base model to predict the correct class from input text sequences. The entire training process is implemented using **PyTorch**.

> [**Model Architecture**]
> Input Text → BERT Tokenizer → BERT Base Model → Classification Head → Class Probabilities

---

https://github.com/user-attachments/assets/dd6fb2f4-afdd-4c62-a678-66fcd01e59a5

---
### **Dataset**
Using the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) news articles  dataset from Hugging Face . 
#### **Dataset Information**
1. **Classes**: 4 (World, Sports, Business, Sci/Tech)
2. **Samples**: 120,000 training, 7,600 test

---
### **Project Workflow**

1. **Configuration Setup**
    - Define model (BERT base), dataset (AG News), and training parameters
    - Set hyperparameters
    - Configure optimization
2. **Data Preparation**    
    - Load dataset from Hugging Face and convert to DataFrame
    - Create label mappings (id2label and label2id)
3. **Dataset Processing**
    - Implement custom PyTorch Dataset class
    - Prepare data for DataLoader with 32-sample batches
4. **Model Architecture**
    - Build BERT classifier with dropout (0.3)
    - Freeze embeddings and first 6 layers (43M frozen params)
    - Train only remaining parameters
5. **Training Setup**
    - Create train/evaluation functions
    - Configure optimizer with warmup

