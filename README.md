# deep-learning-challenge

## Overview of the Analysis

# Purpose of the Analysis

The purpose of this analysis is to develop a deep learning model to classify whether an application will be successful based on features from the dataset provided by Alphabet Soup. 
This report documents the steps taken to preprocess the data, build and optimize the neural network model, and evaluate its performance.

# Results

## Data Preprocessing



*  Target Variable: 

     * IS_SUCCESSFUL: This binary variable indicates whether an application was successful (1) or not (0). It is the target variable for the classification model.

*  Feature Variables: 

     * The features used in the model include all the columns in the dataset after preprocessing. Specifically, these features are:	 
	 
        * APPLICATION_TYPE

        * AFFILIATION
		
		* CLASSIFICATION
		
		* USE_CASE
		
		* ORGANIZATION

        * STATUS
		
		* INCOME_AMT
		
		* SPECIAL_CONSIDERATIONS
		
		* Various other features resulting from one-hot encoding of categorical variables.

*  Variables to Remove:

      * Removed Columns: EIN and NAME, as they are unique identifiers and do not provide meaningful input for the model.
	  
	  * Other Columns: After preprocessing, any columns with less informative data or high cardinality not contributing to the model’s performance were also excluded.


# Compiling, Training, and Evaluating the Model


## Model Architecture:


* Number of Neurons and Layers

    * First Hidden Layer: 500 neurons with ELU activation.
	
	* Second Hidden Layer: 200 neurons with ReLU activation.
	
	* Third Hidden Layer: 128 neurons with ReLU activation.
	
	* Fourth Hidden Layer: 64 neurons with ELU activation.
	
	* Dropout Layer: 50% dropout to reduce overfitting.
	
	* Output Layer: 1 neuron with sigmoid activation to predict the probability of success.


* Activation Functions:

  * ELU (Exponential Linear Unit): Used in the first and fourth hidden layers to improve learning dynamics and avoid dead neurons.
  
  * ReLU (Rectified Linear Unit): Used in the second and third hidden layers for its simplicity and effectiveness in hidden layers.
  
  * Dropout: Applied to prevent overfitting by randomly dropping neurons during training.
  

* Achieving Target Performance:

  * Initial Model Performance: 
     
	  * Initial Accuracy : Approximately 72%.
  
      * Traning Summary : 215/215 - 0s - 1ms/step - accuracy: 0.7249 - loss: 0.5666
	  
      * Loss:  0.5665735602378845, Accuracy: 0.7249271273612976
  
  * Optimized Model Performance: 
  
      * Optimised Accuracy : Approximately 72%.
  
      * Training Summary: 215/215 - 0s - 2ms/step - accuracy: 0.7241 - loss: 0.5580
      
	  * Loss: 0.558035135269165, Accuracy: 0.7240524888038635

* Steps Taken to Increase Model Performance:

   * Increased Neurons and Layers: Added more neurons and hidden layers to capture more complex patterns.
   
   * Changed Activation Functions: Incorporated ELU activation to improve model learning.

   * Batch Normalization: Added to stabilize and accelerate training.
   
   * Dropout: Applied to reduce overfitting and improve generalization.
   
   * Early Stopping: Implemented to halt training when the model stops improving, preventing overfitting.
   
   * Learning Rate Scheduling: Adjusted learning rates to optimize training dynamics.


# Summary


* Overall Results:

   *  The model achieved an accuracy of approximately 73% with the current setup. Despite extensive optimization efforts, including increased model complexity, dropout, batch normalization, 
      and activation function adjustments, the target accuracy of 75% was not reached. Further fine-tuning and experimentation are needed.  
	  
	     
* Recommendation for Alternative Models:

   *  Random Forest Classifier:

         * Why: Random Forests can handle high-dimensional data well and are less prone to overfitting compared to deep learning models. They provide feature importance, which can help in understanding key factors influencing success.
		 
		 
   *  Gradient Boosting Machines (GBM):

         * Why: GBM algorithms can also manage complex patterns in data and often perform well on classification tasks. They are effective in capturing non-linear relationships and interactions between features.

* Explanation: Using ensemble methods like Random Forests or GBMs can potentially offer better performance on this dataset due to their ability to capture intricate patterns without requiring extensive hyperparameter tuning and large data volumes. 
 

## Technologies Used

- Python

- Pandas

- Tensorflow

- scikit-learn

- Jupyter Notebook




## Folder Structure 

The deep-learning-challenge consists of the following folders and files:

* **deep-learning-challenge folder**:

   * Starter_Code.jpynb   -  The Jupyter Notebook file containing the code & analysis. 
   
   * AlphabetSoupCharity_Optimisation.jpynb  - The Jupyter Notebook file for model optimization.
   
   * AlphabetSoupCharity.h5  - The saved model file before optimization.
   
   * AlphabetSoupCharity_Optimisation.h5  - The saved model file after optimization.  

   * README.md**- Provides the Overview of the Analysis and project folder Structure


## How to Use

1. Clone the GitHub repository to your local machine using the following command:

    git clone https://github.com/SriPenumatcha/deep-learning-challenge.git

2. Open the `Starter_Code.jpynb' file using Jupyter Notebook.

3. Run each cell in the notebook to perform the analysis and view the results.

4. Open the `AlphabetSoupCharity_Optimisation.jpynb' file using Jupyter Notebook.

5. Run each cell in the notebook to perform the analysis and view the results.

6. Review the analysis findings and conclusions in the notebook and the README.md file
