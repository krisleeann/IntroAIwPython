import csv
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

# NOTE: Had to create due to debugging issue from typecasting individually with regular variables in the load_data function. I kept receiving Key Errors during compiling.

def visitors(visitor):
    if visitor != "Returning_Visitor":
        return 0
    else:
        return 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)
    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Convert 'Month' from string to int using dict
    month = {'Jan': 0, 'Feb': 1,  'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
             'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    # Convert 'VisitorType' to int
    visitor = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": 0}

    bool_type = {"FALSE": 0, "TRUE": 1}

    # Open csv file and read, return tuple
    with open(filename) as file:
        data = csv.DictReader(file)

        # Initialize evidence and labels as lists
        labels = []
        evidence = []

        for d in data:
            # Typecast each row using native data types or assignments created above, per Ben's suggestion
            conversions = [int(d["Administrative"]),
                           float(d["Administrative_Duration"]),
                           int(d["Informational"]),
                           float(d["Informational_Duration"]),
                           int(d["ProductRelated"]),
                           float(d["ProductRelated_Duration"]),
                           float(d["BounceRates"]),
                           float(d["ExitRates"]),
                           float(d["PageValues"]),
                           float(d["SpecialDay"]),
                           month[d["Month"]],
                           int(d["OperatingSystems"]),
                           int(d["Browser"]),
                           int(d["Region"]),
                           int(d["TrafficType"]),
                           visitor[d["VisitorType"]],
                           bool_type[d["Weekend"]]]
            # Append to end of evidence and labels lists, then return tuple
            evidence.append(conversions)
            labels.append(0 if d["Revenue"] == "FALSE" else 1)
            
            return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Modeled after 'Banknotes' source code, per Reem's suggestion
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).
    Assume each label is either a 1 (positive) or 0 (negative).
    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.
    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = float(0)
    specificity = float(0)

    true_positive = 0
    true_negative = 0

    # Code format and use of zip via Reem's section
    for a, b in zip(labels, predictions):
        if a == 1:
            true_positive += 1
            if a == b:
                # Increment sensitivity if prediction and purchase made were correctly identified
                sensitivity += 1
            elif b == 0:
                true_negative += 1
            if b == a:
                # Increment specificity if prediction and purchase not made were correctly identified
                specificity += 1

    sensitivity = true_positive / labels.count(1)
    specificity = true_negative / list(predictions).count(0)
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
