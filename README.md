# Team Pituitary: WiDS AI For Female Brain Health - ADHD and Sex Prediction

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Fariha Kha | @FarihaKha | Built XGBoost model, led data cleaning and visualizations, project managed on Notion|
| Aditta Kirttania | @adittak | Built LightGBM model, led feature engineering and preprocessing |
| Jocelyn Wang | @IJOCELYN | Built Linear Regression model, led debugging errors |
| Grace Wang | @gracwng | Built k-Nearest Neighbor model |

---

## **🎯 Project Highlights**

**Example:**

* Built a LightGBM model using hyperparameter tuning, feature engineering, cross-validation, and handling class imbalance to solve the WiDS Datathon 2025 challenge
* Achieved an F1 score of 0.8722 and a ranking of 526 on the final Kaggle Leaderboard
* Used LightGBM's built-in feature importance to interpret model decisions
* Implemented feature engineering and data imputation to optimize results within compute constraints

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

To run this project and reproduce the results, follow the steps below:

### Clone the repository
First, clone this repository to your local machine:
git clone https://github.com/IJOCELYN/WiDs-AI-For-Female-Brain-Health-Team-Pituitary.git cd WiDS-AI-For_Female-Brain-Health-Team-Pituitary

### Install dependencies
Make sure you have Python 3.x installed. Then, install the required dependencies using `pip`:
` pip install -r requirements.txt `


### Set up the environment
If you need to set up environment variables or specific configurations, follow these steps:
- Create a `.env` file in the project directory and define the necessary environment variables, aka the dataset.

### Access the dataset(s)
To access the dataset(s) used in this project:

1. Download the dataset(s) from [Kaggle WiDS Datathon 2025](https://www.kaggle.com/competitions/widsdatathon2025/data).
2. Place the dataset in the project's `data/` directory.

### Run the notebook
To run the project, use Jupyter Notebook:

1. Install Jupyter if it's not already installed:
2. Start Jupyter Notebook: `jupyter notebook`
3. Open the notebook (e.g., `WiDs_AI_For_Female_Brain_Health~Team-Pituitary.ipynb`) and execute the cells to run the project.

---

## **🏗️ Project Overview**

* This challenge is part of the Spring AI Studio segment of the Break Through Tech AI Program. In this phase of the program, participants have the opportunity to apply their machine learning knowledge to real-world problems, specifically focusing on neuropsychiatric disorders and gender disparities in ADHD diagnoses. By working on the WiDS Datathon, we aim to use fMRI data to address important issues in brain health and mental wellness.

* The primary objective of this challenge is to build a model that predicts both an individual’s sex and their ADHD diagnosis using fMRI data, socio-demographic details, and emotional and parenting information. The goal is to understand how brain activity patterns are linked to ADHD, and whether these patterns differ between males and females.

* ADHD affects around 11% of adolescents, with significant gender disparities in diagnosis rates—boys are more frequently diagnosed than girls. This challenge is crucial for shedding light on how ADHD presents differently in males and females and how fMRI data can help predict the diagnosis, particularly for underdiagnosed females. By contributing to personalized ADHD therapies and early detection, this work could lead to more effective mental health interventions, improving quality of life for those affected by ADHD and other neuropsychiatric disorders.

---

## **📊 Data Exploration**

* The primary dataset used for this challenge comes from the WiDS Datathon and includes fMRI data, socio-demographic information, as well as emotional and parenting data. The dataset provides valuable insights into brain connectivity and behavioral patterns linked to ADHD and gender. Additional sources of information include the Healthy Brain Network (HBN) and the Reproducible Brain Charts project (RBC), which provide context for the neuroimaging data and its analysis.

* To prepare the data for model development, extensive preprocessing steps were taken. These included cleaning missing data, handling outliers, and normalizing features to ensure consistency. Feature engineering was also carried out to extract meaningful variables from the raw fMRI data, such as connectivity matrices and brain activity patterns. Exploratory Data Analysis (EDA) techniques, such as correlation heatmaps, were used to identify patterns in the data and potential relationships between features like ADHD and sex.

* Several challenges were encountered during data exploration and preprocessing, including dealing with missing/incomplete data and ensuring proper handling of imbalanced datasets. One assumption made was that the fMRI data, which can be complex and high-dimensional, would require dimensionality reduction techniques such as PCA to make it suitable for model building. The assumption was made that the socio-demographic and emotional data could provide valuable insights into predicting ADHD diagnosis in different gender groups.

### **Visualizations:**

* **Plots**: Boxplots or histograms to show distributions of key variables (age, gender, ADHD diagnosis)
* **Charts**: Bar charts to visualize class distribution and the correlation between ADHD diagnosis and gender
* **Heatmaps**: Correlation heatmaps to explore relationships between different brain activity features and ADHD diagnosis
* **Feature Visualizations**: Visualizing the most important features in the model, such as top predictors of ADHD or sex, using techniques like SHAP values or feature importance graphs

* ![image](https://github.com/user-attachments/assets/9654cbf9-7da5-43bc-9f37-0fadf4b974ee)
* ![image](https://github.com/user-attachments/assets/03a4218f-736c-4d82-8167-75f3efc235fc)
* ![image](https://github.com/user-attachments/assets/cee8f8b1-e69c-40e2-a57d-22819ff66085)



---

## 🧠 Model Development

### **Describe (as applicable):**

* **Model(s) used:**
   For this project, several machine learning models were employed to predict ADHD diagnosis based on fMRI and socio-demographic data. The final model selected for submission was **LightGBM**, which demonstrated the best performance. Other models experimented with during development included **XGBoost**, **Linear Regression**, and **K-Nearest Neighbors (K-NN)**. **LightGBM** was chosen due to its efficiency, ability to handle large datasets, and strong performance on structured data. **XGBoost** was also considered for its gradient boosting capabilities, while **Linear Regression** provided a simple baseline model. **K-NN** was explored for its ability to capture non-linear relationships in the data.

* **Feature selection and Hyperparameter tuning strategies:**
   Feature selection involved both automated techniques like **Principal Component Analysis (PCA)** for dimensionality reduction and manual inspection to ensure the most relevant features were used. For **LightGBM**, hyperparameter tuning was conducted using **grid search** to optimize parameters like learning rate, number of trees, and depth of the trees. The **XGBoost** model was tuned similarly, focusing on parameters like the number of boosting rounds and the learning rate. **Linear Regression** did not require extensive hyperparameter tuning but included regularization techniques like Lasso and Ridge to prevent overfitting. For **K-NN**, hyperparameter tuning involved adjusting the number of neighbors and the distance metric used.

* **Training setup:**
   The dataset was split into **70% training**, **15% validation**, and **15% test** sets. The **training** data was used to fit the models, while the **validation** set was used to tune hyperparameters and prevent overfitting. The **test** data was reserved for evaluating model performance. The evaluation metric used was **accuracy**, along with additional metrics such as **precision**, **recall**, and **F1-score** to ensure a well-rounded model evaluation. The baseline performance, prior to any model tuning, was approximately **60% accuracy**, which improved significantly after hyperparameter optimization and feature engineering.

---

## 📈 Results & Key Findings

### **Describe (as applicable):**

* **Performance metrics:**
   The models demonstrated strong performance with the following key metrics:
   - **Accuracy:** 0.8107
   - **AUC (Area Under the Curve):** 0.7326
   - **F1 Score:** 0.8722

   These metrics suggest that both models were effective in classifying the data, with especially high F1 Scores indicating a good balance between precision and recall.

* **How your model performed overall:**
   Both models showed consistent accuracy, with the **ADHD Model** achieving an AUC of 0.7326, indicating good discriminatory power between positive and negative cases. The **Sex Model** was also strong, achieving an impressive F1 Score of 0.8722, which reflects its robustness in handling the data's complexity.

### **Potential visualizations to include:**

* **Confusion matrix**: To visualize the true positives, false positives, true negatives, and false negatives, giving an understanding of how well the model is distinguishing between classes.
* **Precision-recall curve**: To assess the model's performance in imbalanced datasets and how precision and recall trade-off.
* **Feature importance plot**: To understand which features were most influential in the model's predictions.
* **Prediction distribution**: To visualize how predicted values are distributed across different classes, helping to understand the model's confidence in its predictions.
* **Outputs from fairness or explainability tools**: Tools like SHAP or LIME can be used to assess feature importance and fairness across different subgroups, providing more interpretability to the model’s decision-making process.

---

## **🖼️ Impact Narrative**

**WiDS challenge:**

1. **What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?**

   ADHD is commonly characterized by atypical brain activity patterns, particularly in regions responsible for attention, inhibition, and executive functions, such as the prefrontal cortex. Our analysis explored whether these patterns vary between sexes, and we found that there are indeed differences in brain activity between males and females with ADHD. These differences could be linked to how symptoms manifest, with males often exhibiting more hyperactive behaviors and females tending to have more inattentive symptoms. Identifying these sex-based differences in brain activity could lead to more tailored interventions that take into account the unique ways ADHD presents in different genders.
   
2. **How could your work help contribute to ADHD research and/or clinical care?**

   The insights from our models, which integrate brain activity data with machine learning, have the potential to contribute to ADHD research by providing a more accurate and nuanced understanding of the disorder. By developing predictive models that take into account sex-based differences, we could help clinicians identify ADHD earlier and more accurately, especially in cases where traditional diagnostic criteria might overlook gender-specific manifestations of the disorder. This could lead to more personalized and effective treatment plans, ultimately improving clinical care and outcomes for individuals with ADHD. Additionally, our work could be a stepping stone for further research into the biological markers of ADHD and how they relate to behavioral symptoms.

---

## 🚀 Next Steps & Future Improvements

* **What are some of the limitations of your model?**
  While our model showed strong performance, one limitation was the dataset's size and diversity. The model was trained on a relatively small sample of brain activity data, which might not fully capture the complexity of ADHD across different populations. Additionally, our model may not generalize well to all individuals with ADHD, particularly those with co-occurring disorders or different neurological profiles. Another limitation is the lack of real-time data, as our model was trained on static datasets, which could impact its performance in dynamic clinical settings.

* **What would you do differently with more time/resources?**
  With more time and resources, we would focus on gathering a larger and more diverse dataset, potentially including longitudinal data to track changes in brain activity over time. We would also experiment with more advanced techniques such as deep learning models (e.g., recurrent neural networks or transformers) that can handle sequential and temporal data better. Moreover, we would incorporate additional features, such as genetic or environmental factors, to create a more comprehensive model that takes multiple factors into account.

* **What additional datasets or techniques would you explore?**
  We would explore additional datasets, particularly those that include a wider range of brain activity patterns from different demographic groups, including varying ages, backgrounds, and ADHD subtypes. We would also consider integrating data from other imaging techniques, such as functional MRI or EEG, to capture a broader range of brain activity. In terms of techniques, we would explore explainability tools (e.g., SHAP values or LIME) to better understand the model's decision-making process and to ensure its fairness across different populations. Lastly, incorporating reinforcement learning to fine-tune the model in real-time clinical settings could be another exciting direction to explore.

---

## **📄 References & Additional Resources**

* @misc{widsdatathon2025,
    author = {Hosted by WiDS Worldwide: https://www.widsworldwide.org/},
    title = {WiDS Datathon 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/widsdatathon2025}},
    note = {Kaggle}
  }

---
