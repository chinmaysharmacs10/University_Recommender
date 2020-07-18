# MS in US - University Recommender
* Created a deep learning model that recommends a University a student should apply to based on the academic details entered. A web application was made to take in the details of the student and get a recommendation from the model by passing the details as input.
* The model is a multi-class classifier that provies the class (which is the University) based on the details of the applicant.
* The data was cleaned of missing values and more preprocessing was done to obtain the cleaned dataset. 
* Exploratory data analysis on the cleaned data to get insights about the data and its distribution.
* A __deep neural network__ was built using __Keras__ for multi-class classifiaction.
* Built a client facing web application using __Django__. 
* __Python Version__: 3.6.10
* __Libraries Used__: Pandas, Matplotlib, Seaborn, sklearn, imblearn, Tensorflow, Keras, Django.
* References: https://github.com/aditya-sureshkumar/University-Recommendation-System/blob/master/recommendationSystem/Ramkishore_Joe_Swetha_Aditya.pdf

## Cleaning the Data
###### (File: data_cleaning.ipynb)
* The dataset __admission_data.csv__ consists of data from 54 different US Universities about the academic details of the applicants.
* Dropped the redundant columns (userName, userProfileLink, program, toeflEssay, topperCgpa, termAndYear) and columns that had NaN in majority (gmatA, gmatQ, gmatV).
* Removed entries that had earlier format GRE and TOEFL scores to ensure consistency with current pattern.
* Since we have to make a classification model, we need only those entries that got an admit. So kept all entries with admit=1.
* Some of the Universities had very few admits, so kept only universities that had around 100 admits.
* CGPA was on different scales (10-scale, 100-scale, 4-scale, 5-scale). All cgpa entries were converted to 4-scale for uniformity.
* __Feature Importance__ method was used to calculated importance of each feature. greV, greQ, greA, toeflScore, cgpa_4, researchExp, industryExp, internExp & ugCollege were the most important features.

![alt text](https://github.com/chinmaysharmacs10/University_Recommender/blob/master/Images/feature_selection.png "Importance of Features")

* The cleaned data set consists 36 different universities (i.e. classes). (File: 'admission_data_cleaned.csv')

## Exploratory Data Analysis (EDA)
###### (File: data_analysis_EDA.ipynb)
* The ugCollege had 980 categories which will add alot of dimensions to the data after one-hot encoding. Also, a sigle college is written in different ways which causes inconsistency, eg: BIT, Mesra and Birla Institute of Technology, Mesra. So the ugCollege column is removed.
* Generated a word cloud to display the different UG Colleges and their frequency in the dataset.

![alt text](https://github.com/chinmaysharmacs10/University_Recommender/blob/master/Images/wordcloud_ugCollege.png "UG Colleges")

* Number of admits per university was plotted to see the distribution of data in different classes. This distribution suggests that the dataset is imbalanced.

![alt text](https://github.com/chinmaysharmacs10/University_Recommender/blob/master/Images/university_admits.png "Admits per university")

* Plots for average of each parameter vs University were made to see the variation of parameters between different Universities(classes).
* Histograms and boxplots were made for the parameters to see the distribution of thier values and to check for outliers. The plots suggest data needs to be scaled.


## Data Preprocessing
###### (File: classifier_model.py)
* To remove the imbalance in the dataset and prevent high bias in the model, __SMOTE - Synthetic Minority Oversampling Technique__ from __imblearn library__ is used for over sampling with sampling_stratergy set to 'not majority'. It generates new instances for all the classes except the majority class to balance the dataset.  
* The university names we encoded with unique labels by __LabelEncoder__ of __sklearn library__ to be fed to SMOTE.
* __RobustScaler__ of __sklearn library__ was used to scale the data in order to reduce the effect of variation in parameter values, as there is alot of variation in case of researchExp, industryExp & internExp.
* The data was split into train and test set, with 20% of data as test set.

## Building the Multi-class Classification Model
###### (File: classifier_model.py)
* A deep neural network with an input layer (400 neurons), 2 hidden layers (800 & 100 neurons respectively) and an output layer with 36 neurons (for 36 classes) is built using the __Keras (Tensorflow backend)__ library.
* The input and hidden layers have ReLU activation function, and for multiclass classification task the output layer has a Softmax acivation function.
* The model is trained with __Adam__ optimizer and __categorical crossentropy__ loss. It achieves a good __accuracy of 78.64__.
* To visualize the correctness of classification __Confusion Matrix__ is plotted.

![alt text](https://github.com/chinmaysharmacs10/University_Recommender/blob/master/Images/Confusion_Matrix_3.png "Confusion Matrix")

* Model was pickled using the __Joblib library__.






