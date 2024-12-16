# 🧠 Neural Network for MNIST Digit Recognition  

This repository contains a project where I built and trained a custom neural network to predict handwritten digits using the **MNIST dataset** (Modified National Institute of Standards and Technology). This is a foundational project in machine learning and computer vision, showcasing the use of neural networks to perform digit classification.

---

## 📋 **Project Overview**

The goal of this project was to develop a neural network from scratch to accurately predict digits (0–9) from the MNIST dataset. The dataset consists of **42,00 grayscale images** (in the training set) of handwritten digits, each sized **28x28 pixels**.  

**Key Highlights of the Project**:  
- Built a **custom neural network** model from scratch.  
- Achieved digit classification with **high accuracy** on test data.  
- Implemented training, evaluation, and visualization workflows.  

---

## 🚀 **Technologies Used**

- **Programming Language**: Python  
- **Libraries/Frameworks**:  
    - NumPy
    - Pandas (optional, for data management) 
    - Leveraged libraries like **tqdm** for progress visualization during training.  
    - Scipy.special only for the implementation of the logsumexp function.   
    - Matplotlib (for visualization)
    - Json to store the model easily
     

---

## 📂 **Project Structure**

```plaintext
├── dataset/                # MNIST dataset (if locally stored)
├── train.py                # Training script
├── README.md               # This file
└── main.py                 # Used to test the model visually
````
---

## 🧩 **How the Neural Network Works**

- **Data Preprocessing**:  
  - Normalized pixel values to range [0, 1].  
  - Reshaped the data for compatibility with the model.  

- **Model Architecture**:  
  - **Input Layer**: 28x28 neurons (flattened image).  
  - **Hidden Layers**: Fully connected layers with activation functions (ReLU).  
  - **Output Layer**: 10 neurons (for digits 0–9) with a **logsoftmax activation** (chosen for better numerical stability).  

- **Training**:  
  - Trained the model for **14,000 epochs** with a batch size of **100**.  
  - Used a **loss function** and an **optimization function** implementing the **steepest descent** algorithm.
 
---

## 📊 **Results**
The model achieved an accuracy of around 92.5% with only two layers and less than a minute of training (on a pretty decent computer).

