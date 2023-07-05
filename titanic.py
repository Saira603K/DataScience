import pandas as pd
from collections import defaultdict
from google.colab import files
uploaded = files.upload()

'''titanic (1).csv
titanic (1).csv(text/csv) - 44225 bytes, last modified: 7/3/2023 - 100% done
Saving titanic (1).csv to titanic (1).csv'''

import io

class NaiveBayesForCategorical:
    def __init__(self):
        self.class_prob = defaultdict(float)
        self.conditional_prob = defaultdict(lambda: defaultdict(float))

    def fit(self, data, target):
        self.classes = data[target].unique()
        total_count = len(data)
        for class_ in self.classes:
            class_data = data[data[target] == class_]
            self.class_prob[class_] = len(class_data) / total_count
            for feature in data.columns:
                if feature == target: continue
                feature_counts = class_data[feature].value_counts()
                for value, count in feature_counts.items():
                    self.conditional_prob[(class_, feature, value)] = count / len(class_data)

    def predict_proba(self, **kwargs):
        results = {}
        for class_ in self.classes:
            prob = self.class_prob[class_]
            for feature, value in kwargs.items():
                prob *= self.conditional_prob[(class_, feature, value)]
            results[class_] = prob
        return results

# load titanic dataset
df = pd.read_csv(io.BytesIO(uploaded['titanic (1).csv']))

# specify target and features
target = 'Survived'
features = ['Pclass', 'Sex']

# initialize and train model
nb = NaiveBayesForCategorical()
nb.fit(df[features + [target]], target)

# user input and prediction
while True:
    pclass = input("Enter passenger class (1, 2, or 3) or 'quit' to exit: ")
    if pclass.lower() == 'quit':
        print("Exiting the program...")
        break

    sex = input("Enter sex (male or female): ")
    if sex.lower() == 'quit':
        print("Exiting the program...")
        break

    if pclass not in ['1', '2', '3'] or sex not in ['male', 'female']:
        print("Invalid input. Please try again.")
        continue

    probs = nb.predict_proba(Pclass=int(pclass), Sex=sex)
    print(f"For Pclass={pclass}, Sex={sex}, the probabilities are {probs}")


'''OUTPUT

Enter passenger class (1, 2, or 3) or 'quit' to exit: 1
Enter sex (male or female): male
For Pclass=1, Sex=male, the probabilities are {0: 0.07678702564049522, 1: 0.04886700027031125}
Enter passenger class (1, 2, or 3) or 'quit' to exit: 1
Enter sex (male or female): female
For Pclass=1, Sex=female, the probabilities are {0: 0.01340463163120714, 1: 0.10445881709158278}
Enter passenger class (1, 2, or 3) or 'quit' to exit: 2
Enter sex (male or female): male
For Pclass=2, Sex=male, the probabilities are {0: 0.09310426858910047, 1: 0.031260507525860876}
Enter passenger class (1, 2, or 3) or 'quit' to exit: 2
Enter sex (male or female): female
For Pclass=2, Sex=female, the probabilities are {0: 0.01625311585283866, 1: 0.06682291975711545}
Enter passenger class (1, 2, or 3) or 'quit' to exit: 3
Enter sex (male or female): male
For Pclass=3, Sex=male, the probabilities are {0: 0.35322031794627806, 1: 0.04275862523652235}
Enter passenger class (1, 2, or 3) or 'quit' to exit: 3
Enter sex (male or female): female
For Pclass=3, Sex=female, the probabilities are {0: 0.06166130550355285, 1: 0.09140146495513495}
Enter passenger class (1, 2, or 3) or 'quit' to exit: quit
Exiting the program...

'''
