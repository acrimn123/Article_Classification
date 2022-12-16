# Article Classification Using Deep Learning LTSM Neural Network Model

In this project,the usage of deep learning model to categorize the article into its own label. The dataset consist of 5 label namely Sport, Tech, Business, Entertainment and Politics.

## Getting started

Before proceeding to the model development, the data analyses are done based on the section below :-

- Data Loading

  In this section, the CSV files which is in a raw format of URL were loaded using Pandas module ```pandas.read_csv``` which convert into dataframe.

- Data Inspection
   
  From the dataframe loaded, the inspection is done to check for any abnormalities on the dataframe which then will be cleaned in the next section. The method used from ```Pandas``` module:

  ```
  df.info()           # Check for the datatype
  df.describe()       # Check for dataframe summary
  ```
- Data Cleaning

  After the inspection, the feature columns, ```text```, were cleaned by using ```Regex``` function which basically clean the data from any special characters and leave only lower alphabet words.

  ```
  df.isna().sum()         # Check for missing values
  df.duplicated().sum()   # Check for complete duplicates by row
  df.drop_duplicates()    # Drop duplicates row
  re.sub()                # Regex function to remove/ replace special characters
  ```
- Data Preprocessing
  
  The data preprocessing function used to process the data before it was fit to the model. In this step, the feature columns,, ```text```, were convert into number by  using text preprocesing method in Keras call ```Tokeneizer```. The ```text``` also were padded and truncated using pad_sequences() which also came from Keras module. The label columns, ```category```, which were choosen before in feature selection were transform into its corespondding  number using OneHotEncoder. Both feature and label were then split for model training.

  ```
  Tokenizer()            # tranform each word into corresponding vocablary in integer
  pad_sequences()        # padding and truncating , to ensure all the words have same length, by padding with value 0
  .fit_transform         # OneHotEncoder transform the label columns
  train_test_split       # Split the data to (x_train,x_test,y_train_y_test)
  ```
## Model development

In this stage, the model was develop using ```LTSM``` neural network which was provided in ```tensorflow``` module.The model architecture can be seen below:

<p align="center">
  <img src="https://github.com/acrimn123/Article_Classification/blob/main/model.png" />
</p>

## Model training

For the model training, the model, which have been explained above will be compile. In this project, the model was compile using Adam as optimizer, Categorial Entropy for loss and accuracy as metrics. Then, the model was fit with the (x_train,x_test,y_train_y_test) data split. The result of the training can be seen below:-

<p align="center">
  <img src="https://github.com/acrimn123/Article_Classification/blob/main/PNG/Accuracy.png" />
</p>

<p align="center">
  <img src="https://github.com/acrimn123/Article_Classification/blob/main/PNG/Loss.png" />
</p>

## Model evaluation

After completing the model training, the model will need to be evaluate to check if the model prediction is close to the actual label.The predicion were done using Classification report. The performance of the model are describe in classificaton report as below:-   

<p align="center">
  <img src="https://github.com/acrimn123/Article_Classification/blob/main/PNG/Classification_report.png" />
</p>

## Acknowledgements

 Sepcial thanks to [Susan Li](https://github.com/susanli2016) for the [data](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv). 

