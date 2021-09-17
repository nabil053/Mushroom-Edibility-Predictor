# Mushroom-Edibility-Predictor

A decision tree model that predicts whether a mushroom is edible or poisonous based on a set of attributes.

The dataset used to generate the decision tree is called mushrooms.csv and it includes descriptions of hypothetical samples 
corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society 
Field Guide to North American Mushrooms (1981).

Although the dataset has a total of 22 attributes (class excluded), only four attributes (odor, spore-print-color, habitat, 
cap-color) define the structure of the decision tree model.

The dataset used here contains a total of 8124 sample data, which has been split into training data (5686, 70%) for training 
the model and testing data (2438, 30%) for testing the model's metrics.

The file 'Decision Tree.jpg' included in this repository shows the structure of the decision tree.

The generated model has the ability to predict a sample mushroom's edibility with an accuracy of 99.95%, and its precision 
and recall are 100% and 99.92% respectively.
