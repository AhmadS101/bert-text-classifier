# BERT Classifier Model 

## **Project Overview**
This project implements a multi-class text classification model using [**BERT**](https://huggingface.co/google-bert/bert-base-uncased)  as its base architecture. The model adds a **classification head**  ( neural network layer added on top of BERT that performs the final classification) on top of the pre-trained BERT base model to predict the correct class from input text sequences. The entire training process is implemented using **PyTorch**.

> [**Model Architecture**]
> Input Text → BERT Tokenizer → BERT Base Model → Classification Head → Class Probabilities



https://github.com/user-attachments/assets/dd6fb2f4-afdd-4c62-a678-66fcd01e59a5


## **Dataset**
Using the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) news articles  dataset from Hugging Face . 
#### **Dataset Information**
1. **Classes**: 4 (World, Sports, Business, Sci/Tech)
2. **Samples**: 120,000 training, 7,600 test


## **Project Workflow**

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


## **Test Set Results over each class**
<div align="center">
**Model Accuracy:** 94.24%

| Class     | Precision | Recall  | F1-Score |
|-----------|-----------|---------|----------|
| World     | 95.77%    | 95.37%  | 95.57%   |
| Sports    | 98.74%    | 98.74%  | 98.74%   |
| Business  | 92.08%    | 90.00%  | 91.03%   |
| Sci/Tech  | 90.42%    | 92.84%  | 91.61%   |
</div>

---
## **Usage**

#### 1. Clone the repository
```
https://github.com/AhmadS101/bert-text-classifier.git
cd bert-text-classifier
```
#### 2. Create a virtual environment
- **Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```
- **Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```
#### 4. Run the Streamlit app
```python
streamlit run streamlit_app.py
```
#### 5. Train the Model on Your Custom Dataset

> ⚠️ **Note:** To train the BERT text classification model on your own dataset, update the value of `src/constants.DATASET_NAME`.

```python
python training.py
```
## **Project Structure**
```
bert-text-classifier/
├── notebook/
│ └── bert-text-classifier.ipynb 
├── model/
│ └── bert_clf_model.pt 
├── src/
│ ├── constants.py 
│ ├── data_loader.py 
│ ├── model.py 
│ ├── trainer.py 
│ ├── prediction.py 
│ └── helper.py 
├── requirements.txt 
├── training.py 
├── streamlit_app.py 
└── README.md
```