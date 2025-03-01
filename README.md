# MIE1628-Project-2

Part A: Advanced Data Analysis with PySpark or SQL Spark 
1. Using the provided integer.txt file, develop a spark script to count the number of odd and even integers.
  
2. Salary Aggregation with Statistical Analysis
-  Analyze the salary.txt file to compute total salaries per department and investigate trends or discrepancies in salary distributions. 
- Utilize statistical measures (e.g. mean, median, standard deviation) and visualizations of boxplot and histogram.

3. Implement an Optimized MapReduce
-  Utilize the shakespeare.txt file to implement an optimized MapReduce operation that counts specific terms, allowing for case-insensitivity and punctuation removal. 
-  The code is for counting the number of times particular words appear in the document.

4. Word Frequency and Distribution Analysis
- From shakespeare.txt, calculate top 10 and bottom 10 words. Show 10 words with the most count and 10 words with the least count. 

PART B: Advanced Recommender System with Apache Spark 
Develop a sophisticated distributed recommender system using Apache Spark’s capabilities, emphasizing accuracy, efficiency, and comparative evaluations.
Utilize the provided movies.csv dataset.

1. Data Description and Insights Analysis
- Identify the top 10 users who have contributed the most ratings (not just high ratings) and discuss their influence on the dataset.
- Conduct EDA, such as visualizing the distribution of ratings, to uncover patterns in user behaviors and preferences.
- Discuss potential implications for marketing strategies based on user engagement and rating tendencies.
   
2. Split Dataset and Performance Assessment
- Experiment: Split the dataset into training and testing subsets using 2 different ratios (for e.g. 60/40, 70/30, 75/25, and 80/20). Implement stratified sampling to ensure users are represented proportionally across the splits. 
- Report on how different splits influence the performance of your collaborative filtering model (you can use one of the evaluation metrics to show this). 
- Represent the performance variation of the model based on each split ratio, identifying the most effective configuration based on empirical findings.
 
3. In-Depth Evaluation of Error Metrics
- Define and explain key metrics for evaluation: MSE, RMSE and MAE. Introduce advanced metrics Precision, Recall, and F1 Score specifically tailored for recommendations. 
- Provide a detailed evaluation of each model's performance using these metrics, discussing the strengths and weaknesses of each in the context of a recommendation system focused 
solely on user ratings. 
- Make observations about the trade-offs involved when selecting different metrics, particularly in scenarios of sparse data or imbalanced ratings.

4. Hyperparameter Tuning Using Cross-Validation Techniques
- Tuning: Conduct systematic hyperparameter tuning for at least two parameters of the collaborative filtering algorithm, such as rank, regularization, or iterations, etc. utilizing methods like grid search or randomized search combined with cross-validation. 
- Visualize the impact of different hyperparameter configurations on model performance (e.g., varying RMSE scores) and provide a rationale for your tuning choices based on your findings. 
- Discuss how each parameter affects model performance and overall training time, including insights on how to balance complexity against performance.

5. Personalized Recommendations and Analysis for Selected Users 
- Recommendations: Generate personalized movie recommendations for user IDs 11 and 21 based on their rating preferences. 
- Discuss how the collaborative filtering approach utilizes user ratings to generate these recommendations and the effectiveness of this technique for a dataset with limited features. 
- Compare performance outcomes between the refined recommendations for these users and any baseline recommendations generated earlier in your analysis. Discuss potential enhancements or features that could be added for improved personalization in future iterations. 
