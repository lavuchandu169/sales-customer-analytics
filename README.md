Sales and Customer Analytics
Overview
This project presents a comprehensive analysis of an augmented retail dataset. The analysis covers various stages, including data preparation, exploratory data analysis (EDA), feature engineering, and predictive modeling using machine learning algorithms. The primary goal of this project is to predict SalesAmount using advanced machine learning models and to evaluate their performance.

Project Structure
The project is divided into the following sections:

Data Loading and Inspection:

The dataset was loaded and inspected to understand its structure, identify missing values, and check for any anomalies.
Basic statistics were generated to get an initial understanding of the data.
Data Preparation:

Handling of missing values (though not applicable for this dataset).
Feature Engineering: Creation of new features such as PricePerUnit, ProfitMargin, and DiscountRatio.
Outlier Detection: Outliers were detected and removed from the SalesAmount column using the Interquartile Range (IQR) method.
Data Scaling: Numerical features were standardized for improved model performance.
Exploratory Data Analysis (EDA):

Visualized the distribution of SalesAmount using histograms.
Analyzed sales trends over time using time series plots.
Compared sales across different product categories using bar plots.
Feature Selection and Model Building:

Selected features like QuantitySold, Discount, Price, Year, and Quarter for modeling.
Built and trained two machine learning models:
Random Forest: A robust model but struggled with accuracy in this case.
Gradient Boosting: Known for combining weaker models, but also struggled with predictive accuracy.
Visualization of Model Performance:

Visualized the actual vs. predicted sales for both Random Forest and Gradient Boosting models.
Compared the performance of both models using scatter plots.
Comparative Evaluation:

Compared both models based on evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²).
Both models underperformed, with negative R-squared values indicating poor predictive power.
Conclusion:

The analysis provided insights into the challenges of modeling retail sales data.
Suggested future work focusing on enhanced feature engineering, alternative modeling approaches, and incorporating domain knowledge for better predictive accuracy.
Files in the Repository
Data_Visualization_Report.docx: Detailed report on the data analysis and modeling process.
Augmented_Retail_Data_Set_No_Time.xlsx: The dataset used for analysis.
notebook.ipynb: Jupyter notebook containing the code for data loading, preparation, analysis, and model building.
README.md: This file.
