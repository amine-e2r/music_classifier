## Project Description

This project implements a music classifier in Python using logistic regression and stochastic gradient descent. It includes a main script (`projet.py`) for training and classifying musical data, as well as a report (`report.pdf`) detailing the methodology and results.  

---

## Key Results

- **Accuracy:** 0.90  
- **Confusion Matrix:**  

|              | Predicted 1 | Predicted 0 |
|--------------|-----------------|-----------------|
| **Actual 1** | 1804            | 74              |
| **Actual 0** | 291             | 1561            |

- The model correctly classified the majority of examples, achieving 90% accuracy.  
- Most misclassifications occur in class 1 (291 false positive), suggesting this class is slightly harder to distinguish.  
- Overall, logistic regression with stochastic gradient descent provides a strong baseline for music classification, with potential improvements in reducing errors for the more challenging class.
