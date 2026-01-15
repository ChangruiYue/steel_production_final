# Report of steel_production by Changrui Yue

作者: Changrui Yue
日期: 2026-01-15

-- Table of Contents --
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Data Description and Exploratory Analysis](#data-description-and-exploratory-analysis)
  - [Dataset Overview](#dataset-overview)
  - [Feature Relationships and Distributions](#feature-relationships-and-distributions)
- [Methodology](#methodology)
  - [Model Selection](#model-selection)
  - [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
  - [Learning Behavior Analysis](#learning-behavior-analysis)
  - [Prediction Accuracy and Residual Analysis](#prediction-accuracy-and-residual-analysis)
  - [Comparative Residual Diagnostics](#comparative-residual-diagnostics)
  - [Quantitative Performance and Efficiency Comparison](#quantitative-performance-and-efficiency-comparison)
- [Discussion](#discussion)
- [Conclusion](#conclusion)

---

## Abstract
This report presents a systematic machine learning analysis for predicting steel production quality based on normalized process data. A complete experimental pipeline is constructed, including exploratory data analysis, preprocessing, model training, learning behavior investigation, and comprehensive evaluation. Four regression models—Random Forest (RF), Support Vector Regression (SVR), Multi-Layer Perceptron (MLP), and Gaussian Process Regression (GPR)—are compared using quantitative metrics, learning curves, and residual-based visual diagnostics. The results demonstrate that Random Forest achieves the most reliable balance between predictive accuracy, robustness, and computational efficiency. Visual analyses further reveal model-specific error patterns and explain observed performance differences.

---

## Introduction
Accurate prediction of steel production quality is critical for improving process stability and reducing operational costs. However, the underlying production process is governed by complex nonlinear interactions among multiple parameters, which limits the effectiveness of traditional linear modeling approaches. With the availability of high-dimensional process monitoring data, machine learning methods provide a promising alternative for modeling such complex relationships. This study aims to evaluate multiple regression models for steel quality prediction, emphasizing not only predictive accuracy but also generalization behavior, error characteristics, and practical feasibility.

---

## Data Description and Exploratory Analysis

### Dataset Overview
The dataset consists of 21 normalized input variables representing steel production process parameters and one continuous output variable corresponding to production quality. The dataset is divided into training, validation, and test sets to ensure unbiased performance evaluation.

### Feature Relationships and Distributions
Figure 1 illustrates the pairwise scatter plots between representative input features and the output variable. Several inputs exhibit discrete or clustered patterns, indicating that certain process parameters operate within limited or predefined ranges. Nonlinear and heterogeneous relationships between inputs and output can also be observed, suggesting that linear regression models would be insufficient for this task. The output distribution shown along the diagonal reveals a concentration around mid-range values, with relatively fewer extreme cases. This imbalance implies that extreme quality conditions may be more difficult to predict accurately.

---

## Methodology

### Model Selection
Four regression models with distinct learning mechanisms are evaluated:
- Random Forest (RF): Ensemble-based model capturing nonlinear interactions and reducing variance
- Support Vector Regression (SVR): Kernel-based nonlinear regression
- Multi-Layer Perceptron (MLP): Feedforward neural network
- Gaussian Process Regression (GPR): Probabilistic nonlinear regression model

### Training and Evaluation
Model hyperparameters are optimized using grid search with cross-validation on the training set. Performance is evaluated using RMSE, MAE, and the coefficient of determination ((R^2)). Computational efficiency is assessed using training and inference time. Exploratory data analysis further reveals that several input features exhibit discrete or clustered distributions, indicating that certain process parameters operate within fixed or limited ranges. Pairwise visual inspection shows heterogeneous and nonlinear relationships between input variables and the output, suggesting that linear modeling approaches may be insufficient. In addition, the presence of skewed distributions and mild outliers implies potential measurement noise or abnormal operating conditions. Instead of removing these samples entirely, robust preprocessing strategies were applied to mitigate their influence while preserving informative patterns. These characteristics motivate the adoption of nonlinear and ensemble-based learning models in this study.

---

## Results

### Learning Behavior Analysis
Figure 2 presents the learning curve illustrating the impact of training data size on model performance. As the number of training samples increases, validation RMSE decreases steadily, while training RMSE increases slightly. This pattern indicates improved generalization and suggests that the model benefits from additional data without suffering from severe overfitting. The diminishing performance gain at larger sample sizes implies that the model is approaching its learning capacity under the current feature set.

### Prediction Accuracy and Residual Analysis
Figure 3 shows the prediction-versus-actual relationship for the Random Forest model. Most predictions align closely with the ideal diagonal line, indicating accurate estimation across the majority of the output range. The corresponding residual distribution in Figure 4 is centered near zero with relatively symmetric dispersion, demonstrating low systematic bias and stable error behavior.

### Comparative Residual Diagnostics
To further investigate model-specific error characteristics, residual distributions for GPR and SVR are presented in Figures 5 and 6. The GPR residuals (Figure 5) exhibit structured patterns and increased variance for larger residual values. This behavior suggests a smoothing effect inherent to Gaussian processes, leading to underestimation of extreme output values. In contrast, the SVR residuals (Figure 6) show heavier tails, indicating higher sensitivity to local data density and kernel parameterization. These characteristics explain why SVR achieves moderate average performance but displays reduced robustness compared to Random Forest.

### Quantitative Performance and Efficiency Comparison
Table 1 summarizes predictive performance on the test set. Random Forest achieves the lowest RMSE and MAE and the highest (R^2), indicating superior predictive accuracy. Model performance on the test set:
- Random Forest: RMSE 0.0671, MAE 0.0489, R^2 0.4472
- SVR: RMSE 0.0719, MAE 0.0536, R^2 0.3649
- GPR: RMSE 0.0736, MAE 0.0556, R^2 0.3343
- MLP: RMSE 0.0755, MAE 0.0571, R^2 0.3002

Table 2 reports training and inference time. Although MLP offers the fastest inference speed, its weaker accuracy limits its applicability. GPR incurs prohibitively high training cost. Random Forest provides the most favorable trade-off between accuracy and efficiency.

| Model         | RMSE  | MAE   | R^2  | Training Time (s) | Inference Time (s) |
|---------------|------:|------:|------:|------------------:|-------------------:|
| Random Forest | 0.0671 | 0.0489 | 0.4472 | 2.93 | 0.081 |
| SVR           | 0.0719 | 0.0536 | 0.3649 | 4.65 | 0.691 |
| GPR           | 0.0736 | 0.0556 | 0.3343 | 150.18 | 0.268 |
| MLP           | 0.0755 | 0.0571 | 0.3002 | 0.90 | 0.0034 |

### Discussion
Figure 7 Model Performance Comparison with Error Bars consolidates the performance comparison of four regression models for steel production quality prediction, with uncertainty represented by the error bars. Among them, Random Forest (RF) consistently emerges as the most suitable choice when balancing accuracy, robustness, and computational efficiency. RF achieves RMSE = 0.0671, MAE = 0.0489, and R^2 = 0.4472, and these advantages persist across different operating regimes, as visualized in the right-hand panel of the figure. This pattern can be attributed to RF’s ensemble structure, which captures nonlinear feature interactions while dampening the impact of noise and outliers, thereby producing more stable residuals and better generalization to unseen data.

By contrast, Gaussian Process Regression (GPR) shows a smoothing tendency that reduces variance but underestimates extreme outputs under high-variance conditions, and its substantial computational cost reduces practicality for iterative model updates in a production setting (RMSE = 0.0736, MAE = 0.0556, R^2 = 0.3343). Support Vector Regression (SVR) sits between RF and GPR in performance; it delivers reasonable accuracy but is notably sensitive to kernel choice and data distribution, leading to residuals that vary across operating regimes (RMSE = 0.0719, MAE = 0.0536, R^2 = 0.3649). Neural networks, exemplified by the Multi-layer Perceptron (MLP), offer fast inference but show higher variance and weaker robustness under the current configuration (RMSE = 0.0755, MAE = 0.0571, R^2 = 0.3002). These results suggest that MLP’s potential is contingent on more data or more careful architectural tuning.

Beyond aggregate metrics, the residual analyses reveal region-specific errors: prediction errors rise in certain operating regimes, indicating that steel quality is shaped by interactions not fully captured by the present feature set. This observation motivates incorporating domain-specific variables (e.g., process conditions, seasonality, exogenous inputs) and applying explainable ML techniques to diagnose and mitigate these gaps. From an industrial perspective, robustness and stability take precedence over marginal gains in accuracy. RF offers the best balance among accuracy, generalization, and computational cost, making it the most suitable candidate for practical deployment. Where interpretability or understanding of drivers is critical, RF can be paired with explainability tools (e.g., SHAP) to illuminate feature contributions, while targeted feature engineering could help close gaps observed in extreme operating regions.

The current results endorse RF as the primary modeling choice for steel production quality prediction. The underperformance of GPR is tied to smoothing and cost considerations, SVR provides a reasonable baseline with kernel-sensitive behavior, and MLPs, although attractive for speed, require more data or architectural optimization to realize their potential. Future work should focus on enriching the feature space with domain-informed variables and adopting explainable modeling approaches to further improve interpretability and robustness, especially in regimes where residuals are largest.

The results consistently indicate that Random Forest is the most suitable model for steel production quality prediction. Its ensemble structure effectively captures nonlinear feature interactions while mitigating the impact of noise and outliers, which explains its stable residual behavior and strong generalization performance. The inferior performance of GPR can be attributed to its smoothing tendency and high computational complexity, which limit its ability to adapt to extreme operating conditions. SVR demonstrates reasonable predictive capability but remains sensitive to kernel selection and data distribution. MLP, while computationally efficient, shows higher variance and weaker robustness under the current configuration. In addition to overall performance metrics, it is important to interpret the observed model behaviors and their limitations. The Gaussian Process Regression model exhibits a smoothing tendency, which leads to systematic underestimation of extreme output values. This behavior is consistent with the probabilistic nature of Gaussian processes and their sensitivity to high-dimensional feature spaces, resulting in structured residual patterns and reduced adaptability.

Support Vector Regression demonstrates moderate predictive accuracy; however, its residual distribution shows heavier tails. This indicates sensitivity to local data density and kernel parameterization, suggesting that SVR performance is strongly influenced by hyperparameter selection and may vary across operating regimes. Although neural network–based models such as MLP offer fast inference speed, their learning curves indicate higher variance and mild overfitting under the current data configuration. This suggests that additional data or architectural optimization would be required to fully exploit their representation capacity.

Furthermore, residual analysis reveals that prediction errors increase in certain operating regions, implying that steel production quality is influenced by complex interactions not fully captured by the available input features. This highlights the importance of incorporating domain-specific variables or explainable learning techniques in future work to enhance model interpretability and robustness. From an industrial perspective, these findings suggest that model selection should prioritize robustness and stability over marginal performance gains. The Random Forest model achieves the most reliable balance between accuracy, generalization, and computational efficiency, making it the most suitable candidate for practical deployment. Residual analyses reveal that prediction errors increase in certain operating regimes, suggesting that steel quality is influenced by complex interactions not fully captured by the available features. Incorporating additional domain-specific variables or explainable modeling techniques may further improve performance.

---

## Discussion
From an industrial perspective, these findings highlight the importance of prioritizing model stability and interpretability over marginal performance gains. The Random Forest model provides the most reliable balance between accuracy, robustness, and computational efficiency, making it the most suitable candidate for practical deployment.

Residual analyses reveal that prediction errors increase in certain operating regimes, suggesting that steel quality is influenced by complex interactions not fully captured by the available features. Incorporating additional domain-specific variables or explainable modeling techniques may further improve performance.

---

## Conclusion
The comparative results indicate that Random Forest consistently outperforms other models across multiple evaluation metrics. This advantage can be attributed to its ensemble structure, which effectively captures nonlinear interactions among features while reducing variance caused by noise and outliers. Gaussian Process Regression demonstrates a smoothing tendency, leading to systematic underestimation of extreme output values. This behavior is reflected in its structured residual patterns and can be explained by the probabilistic nature of Gaussian processes in high-dimensional spaces. Support Vector Regression achieves moderate accuracy; however, its residual distribution exhibits heavier tails, indicating sensitivity to kernel selection and local data density. This suggests that SVR performance may vary across different operating regimes. Although the Multi-Layer Perceptron model offers fast inference speed, its learning behavior shows higher variance under the current configuration, implying that additional data or architectural tuning would be required for improved robustness.

From an industrial perspective, these findings highlight the importance of prioritizing model stability and interpretability over marginal performance gains. The Random Forest model provides the most reliable balance between accuracy, robustness, and computational efficiency, making it the most suitable candidate for practical deployment.

