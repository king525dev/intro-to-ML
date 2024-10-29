---
---

# Python for Data Science and Machine Learning: From Beginner to Expert

### Phase 1: Python Syntax Essentials
**Goal**: Gain a solid understanding of Python basics, which are essential for any data science or ML project.

#### 1. Python Basics
- **Topics**: Variables, data types, operators, conditional statements, loops.
- **Key Libraries**: Practice without libraries initially, using only core Python.

**Mini-Project**: Simple Calculator  
Build a basic calculator that performs addition, subtraction, multiplication, and division.

```python
def calculator(num1, num2, operation):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    else:
        return "Invalid operation"

print(calculator(10, 5, "add"))
```

#### 2. Functions and Modules
- **Topics**: Defining functions, arguments, return values, imports, and modules.

**Mini-Project**: Simple Banking System  
Write functions to simulate depositing, withdrawing, and checking balance.

```python
balance = 0

def deposit(amount):
    global balance
    balance += amount
    return balance

def withdraw(amount):
    global balance
    balance -= amount
    return balance if balance >= 0 else "Insufficient funds"
```

---

### Phase 2: Intermediate Python and Data Science Libraries
**Goal**: Become proficient with libraries like NumPy, Pandas, and Matplotlib.

#### 1. Introduction to NumPy
- **Topics**: Arrays, array indexing, reshaping, basic operations.

**Mini-Project**: Create an Array-Based Calculator  
Build functions to perform basic mathematical operations on arrays.

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

def array_operations(arr1, arr2):
    return arr1 + arr2, arr1 - arr2, arr1 * arr2

print(array_operations(arr1, arr2))
```

#### 2. Introduction to Pandas
- **Topics**: DataFrames, indexing, filtering, merging, aggregation, handling missing data.

**Mini-Project**: Data Cleaning and Analysis  
Use Pandas to load a CSV dataset, clean missing data, and perform basic analysis.

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', None], 'Age': [24, 27, None, 22]}
df = pd.DataFrame(data)
df.fillna("Unknown", inplace=True)
print(df.describe())
```

#### 3. Data Visualization with Matplotlib and Seaborn
- **Topics**: Basic plots (histogram, scatter, bar), customizing plots, and understanding Seaborn for statistical plots.

**Mini-Project**: Visualize a Dataset  
Load a dataset, and create plots for different variables to understand the data distribution.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
tips = sns.load_dataset("tips")
sns.histplot(tips["total_bill"])
plt.show()
```

---

### Phase 3: Introduction to Machine Learning
**Goal**: Learn the basics of machine learning using Scikit-Learn and build foundational models.

#### 1. Data Preprocessing with Scikit-Learn
- **Topics**: Data splitting, scaling, encoding categorical data, and feature engineering.

**Mini-Project**: Data Preprocessing Pipeline  
Use Scikit-Learn to preprocess a dataset, including handling missing values, scaling, and encoding.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Sample dataset
data = [[10, None], [15, 20], [None, 30]]
imputer = SimpleImputer(strategy="mean")
scaled_data = StandardScaler().fit_transform(imputer.fit_transform(data))
print(scaled_data)
```

#### 2. Basic Machine Learning Models
- **Topics**: Linear regression, decision trees, and model evaluation (accuracy, precision, recall).

**Mini-Project**: Simple Classification Model  
Use Scikit-Learn to build a classifier for binary classification (e.g., predict whether someone buys a product).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample data
X = [[0, 0], [1, 1]]
y = [0, 1]

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict(X)
print("Accuracy:", accuracy_score(y, predictions))
```

#### 3. Kaggle and Competition Datasets
- **Topics**: How to use Kaggle, download datasets, and start participating in beginner-level competitions.

**Mini-Project**: Titanic Dataset Exploration  
Load the Titanic dataset, preprocess it, and create a basic classification model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
df = pd.read_csv("titanic.csv")
df.fillna(df.mean(), inplace=True)

# Basic model
X = df[["Pclass", "Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

### Phase 4: Advanced Machine Learning and Model Optimization
**Goal**: Develop skills to fine-tune models, implement feature engineering, and improve accuracy.

#### 1. Feature Engineering and Hyperparameter Tuning
- **Topics**: Feature selection, hyperparameter tuning (GridSearchCV, RandomizedSearchCV).

**Mini-Project**: Model Tuning with GridSearchCV  
Use GridSearchCV to tune hyperparameters of a RandomForest model on the Titanic dataset.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {"n_estimators": [50, 100], "max_depth": [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
```

#### 2. Exploring Ensemble Methods
- **Topics**: Boosting (e.g., XGBoost), bagging, and voting classifiers.

**Mini-Project**: Ensemble Learning with Voting Classifier  
Combine multiple models (e.g., Random Forest, Logistic Regression) for improved predictions.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = VotingClassifier(estimators=[
    ("lr", LogisticRegression()), 
    ("rf", RandomForestClassifier()), 
    ("dt", DecisionTreeClassifier())
])
model.fit(X_train, y_train)
```

---

### Phase 5: Introduction to Deep Learning and NLP
**Goal**: Learn fundamentals of deep learning and NLP, and build simple neural networks.

#### 1. Deep Learning with PyTorch or TensorFlow
- **Topics**: Tensors, basic neural networks, backpropagation, and optimization.

**Mini-Project**: Build a Neural Network for Image Classification  
Create a simple neural network with PyTorch or TensorFlow for MNIST digit classification.

```python
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

#### 2. Natural Language Processing and LLMs
- **Topics**: Text processing, word embeddings, transformer basics, fine-tuning pre-trained models.

**Mini-Project**: Sentiment Analysis with Pre-trained Models  
Fine-tune BERT or DistilBERT on a sentiment analysis dataset using Hugging Face Transformers.

```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
training_args = TrainingArguments(output_dir="./results", per_device_train_batch_size=8, num_train_epochs=2)
trainer = Trainer(model=model, args=training_args, train_dataset=your_dataset)

trainer.train()
```

---
---