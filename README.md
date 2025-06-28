# MLP Implementations: From Scratch & with PyTorch

This repository contains homework assignments for CENG499 - Introduction to Machine Learning at METU, showcasing different implementations of Multi-Layer Perceptrons (MLPs) for both classification and regression tasks.

## 📂 Project Structure

- `part1/`: MLPs implemented from scratch using NumPy.
  - `mlp_classification_backpropagation.py`: MLP classifier using softmax and cross-entropy loss.
  - `mlp_regression_backpropagation.py`: MLP regressor using identity output and MSE loss.
- `part2/`: MLPs implemented with PyTorch.
  - `part2_mlpclassification.py`: MLP with manual parameter initialization, softmax output, and classification.
  - `part2_mlpregression.py`: Regression MLP with tanh activation and MSE loss.
- `part3/`: Hyperparameter tuning and model selection using PyTorch.
  - `part3.py`: Full grid search over activation functions, batch sizes, hidden layer sizes, and epochs.
  - Evaluates models based on accuracy and confidence intervals.

## 📊 Results

- Best test accuracy: **87.55%** using configuration `(128,)` hidden size, `Tanh` activation, `0.001` learning rate, `32` batch size.
- Used **confidence intervals** to report reliable accuracy measurements.
- Plots generated for Structural Risk Minimization (SRM) across configurations.

## 🧠 Key Concepts Demonstrated

- Manual implementation of forward and backward passes
- Gradient descent and weight updates without high-level frameworks
- Deep learning model training using PyTorch
- Hyperparameter tuning and model selection
- Accuracy/loss tracking and visualization
- Confidence interval analysis

## 🔧 Technologies

- Python (NumPy, PyTorch)
- Matplotlib
- Pickle (for dataset handling)

## 🧾 Reports

Each part includes a detailed PDF report explaining:
- Model architecture
- Forward/backward pass logic
- Experimental results and evaluation

## 👨‍💻 Author

**Selim Tarık ARI**  
BSc Computer Engineering – Middle East Technical University  
Email: [seltarari@gmail.com](mailto:seltarari@gmail.com)  
LinkedIn: [linkedin.com/in/selim-tarik-ari](https://www.linkedin.com/in/selim-tarik-ari)  
GitHub: [SelTarARI](https://github.com/SelTarARI)

---

