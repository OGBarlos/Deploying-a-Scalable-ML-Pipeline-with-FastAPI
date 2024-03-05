# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Created model using Random Frest Classifier on the census data

## Intended Use

Predict if a person earns over $50,000 or not based on the census data

## Training Data

This data was collected from the Census Bureau and contains information like: workclass, education, martial status, occupation, relationship, race, sex, native country, age, capital gains/losses, hours per week worked, and education number.

## Evaluation Data
Evaluation Data was 20% of the full data, this was done with test_size in train_model.py to select .2 from the entire set.

## Metrics
Precision = 0.7443, Recall = 0.6390, F1 = 0.6872

## Ethical Considerations
Since this data is public and by the Census Bereau, there should be none or limited bias towards a specific group or class of people.

## Caveats and Recommendations
Since it's older, it doesn't necessarily have any implications about current predictions, so should not be used to make real time predictions/ Therefore this Machine Learning model should be used for training. 