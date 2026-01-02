# Explainable Customer Segmentation for Retail
### A Multi-Modal Framework using Sentiment Analysis, Spatial Clustering, and SHAP

---

### ► Project Overview

This project develops an end-to-end machine learning framework to move beyond traditional, static customer segmentation. By integrating transactional data with unstructured customer feedback (sentiment) and geographic location, this pipeline creates rich, interpretable customer personas. The goal is to provide actionable insights that can drive personalized marketing strategies, improve customer retention, and justify investments in customer experience by directly linking sentiment to monetary value.

This framework addresses the key limitations of traditional models by providing **behavioral depth**, **dynamic data integration**, and **explainability**.

---

### ► Key Technologies & Libraries

*   **Data Manipulation:** Pandas, NumPy
*   **Sentiment Analysis:** Hugging Face Transformers (DistilBERT), PyTorch
*   **Clustering:** UMAP (dimensionality reduction), HDBSCAN (density-based clustering)
*   **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations), Scikit-learn (RandomForest)
*   **Visualization:** Matplotlib, Seaborn, Plotly Express
*   **Geospatial Analysis:** GeoPandas

---

### ► Methodology Pipeline

The project follows a four-stage pipeline to transform raw data into actionable, interpretable segments:

1.  **Sentiment Feature Engineering:** A **DistilBERT** model was fine-tuned on customer reviews to generate a `Positive_Prob` score, converting qualitative feedback into a quantitative feature with an **F1-score of 0.86**.
2.  **Advanced Clustering:** Customer data (transactional, demographic, and sentiment) was processed with **UMAP** for dimensionality reduction and then clustered using **HDBSCAN** to identify density-based segments of varying shapes and sizes.
3.  **Proxy Modeling for Explainability:** A **RandomForest Regressor** was trained to predict `Total Amount` spent, serving as a proxy for customer value. This supervised model forms the basis for the explainability analysis.
4.  **SHAP Analysis:** The trained RandomForest model was explained using **SHAP** to identify and quantify the key drivers of customer spending, making the model's decision-making process transparent.
5.  **Spatial Analysis:** Customer sentiment data was aggregated by state and visualized on a choropleth map to identify geographic patterns.

---

### ► Key Results and Insights

#### 1. Sentiment is a Significant Driver of Customer Spending

The SHAP analysis revealed that after purchase frequency (`Total_Purchases`), **customer sentiment (`Positive_Prob`) is the second most important driver of customer spending**. High positive sentiment consistently increases the predicted monetary value of a customer, providing a direct, data-driven link between customer satisfaction and revenue. `Age` was found to have a negligible impact.

![SHAP Summary Plot](results/shap_customer_value_drivers.png)

#### 2. Geographic Hotspots of Customer Satisfaction

The spatial analysis showed that customer sentiment is not uniform across the country. States like **Delaware and Utah** emerged as "hotspots" with high average sentiment, suggesting that regional factors may be influencing customer experience and providing an opportunity for targeted marketing or logistical improvements.

![State Sentiment Map](results/state_sentiment_map.png)

---

### ► How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YourUsername]/Explainable-Customer-Segmentation.git
    cd Explainable-Customer-Segmentation
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the notebooks:**
    *   The full data processing and model training workflow is in `notebooks/01_Model_Training_Pipeline.ipynb`.
    *   The final analysis and visualization generation is in `notebooks/02_Experiment_Analysis.ipynb`.

---

### ► Future Work

*   **Longitudinal Analysis:** Track segment migration over time to build predictive models for churn risk and customer lifetime value (CLV).
*   **Aspect-Based Sentiment Analysis (ABSA):** Enhance the sentiment model to identify *what specific products or services* are driving feedback.
