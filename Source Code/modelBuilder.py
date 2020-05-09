# %% Imports
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
import matplotlib.pyplot as plt
from pyspark.ml.regression import RandomForestRegressor
import os
from pyspark.ml.regression import LinearRegression as LinReg
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import rand
from sklearn.feature_selection import SelectPercentile, f_regression
from fpdf import FPDF
import tkinter as tk
import tkinter.ttk as tkk
from ttkthemes import ThemedStyle
from tkinter import *
from tkinter import filedialog, GROOVE
import pandas as pd
import threading
import gc
import sys

# %% UI
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main():
    # Starts the model builder in the second thread
    def start_model_builder(root, model_label, label_label, folds_label, percentile_bar, csv_label, model_path_label):
        model_name = model_label.get()
        label_name = label_label.get()
        folds = folds_label.get()
        percentile = round(percentile_bar.get(), 0)
        csv_path = csv_label.cget("text")
        model_path = model_path_label.cget("text")
        model_builder(csv_path, label_name, percentile, model_name, model_path, folds, root)

    # Updates percentage for number of features to keep on screen
    def onScale(val):
        v = round(float(val), 0)
        lvar.set(v)

    # Gets the CSV File path from the UI
    def getCSV():
        csv_file_path = filedialog.askopenfilename()
        pathlabel.config(text=csv_file_path)

    # Gets the dir path for where to store the model
    def exportpath():
        model_paths = filedialog.askdirectory()
        exportp.config(text=model_paths)

    # Resets all the input fields in the UI
    def resets():
        model_label_entry.delete(0, "end")
        entry2.delete(0, "end")
        libscale4.set(0)
        spin.set(0)
        pathlabel.config(text=" ")
        exportp.config(text=" ")

    def loadmo():
        # Start the thread for creating the model
        th.start()
        bar.step(5)
        bar.config(mode='indeterminate')
        bar.start()

    # Create the root
    root = tk.Tk()
    # Set title of the UI
    root.title("Model Builder")

    # Select theme for the UI
    style = ThemedStyle(root)
    style.set_theme("scidgreen")

    # Define the canvas which all items will be added to
    backg = tk.Canvas(root, width=490, height=470, bg='black', relief='groove')
    backg.pack()

    # Loads in background image
    filename = PhotoImage(file=resource_path("src\\bg.png"))
    background_label = Label(root, image=filename)

    # Places background image
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Creates the label for the header and places it in the UI
    headlabel = tkk.Label(root, text='Model Builder', borderwidth=2, width=50)
    headlabel.config(font=('system', 25, "bold"), anchor=CENTER, relief='ridge')
    backg.create_window(250, 50, window=headlabel)

    # Defines the entry for model name
    model_label_entry = tkk.Entry(root, width=20)
    backg.create_window(335, 240, window=model_label_entry)
    lb1entry = tkk.Label(root, text='Model Name:', width=30, borderwidth=1)
    lb1entry.config(font=('Arial', 10))
    backg.create_window(150, 240, window=lb1entry)

    # Defines the entry for label name
    entry2 = tkk.Entry(root, width=20)
    backg.create_window(335, 270, window=entry2)
    lb2entry = tkk.Label(root, text='Input the Target/Label:', width=30, borderwidth=1)
    lb2entry.config(font=('Arial', 10))
    backg.create_window(150, 270, window=lb2entry)

    # Input field for number of folds
    spin = tkk.Spinbox(root, from_=0, to=100, width=10)
    backg.create_window(310, 300, window=spin)
    lb3entry = tkk.Label(root, text='K-Fold Cross Validation:', width=30, borderwidth=1)
    lb3entry.config(font=('Arial', 10))
    backg.create_window(150, 300, window=lb3entry)
    lb3entrys = tkk.Label(root, text='Folds', width=5, borderwidth=1)
    lb3entrys.config(font=('Arial', 10), relief=FLAT)
    backg.create_window(376, 300, window=lb3entrys)

    # Creates slider for UI for selecting % of features to keep
    libscale4 = tkk.Scale(root, from_=0, to=100, orient=HORIZONTAL, command=onScale)
    backg.create_window(335, 330, window=libscale4)
    lvar = tk.IntVar()
    label = tk.Label(root, text=0, textvariable=lvar)
    backg.create_window(340, 350, window=label)
    lb4entry = tkk.Label(root, text='Percentage Of Features To Keep:', width=30, borderwidth=1)
    lb4entry.config(font=('Arial', 10))
    backg.create_window(150, 330, window=lb4entry)

    # Creates a button for browsing for the CSV FIle
    browseButton_CSV = tkk.Button(root, text="Import CSV", command=getCSV, width=20)
    backg.create_window(335, 100, window=browseButton_CSV, )

    # Creates label for csv browsing
    labelCSV = tkk.Label(root, text='Select The CSV File:', width=20)
    labelCSV.config(font=('Arial', 10))
    backg.create_window(150, 100, window=labelCSV)
    pathlabel = tk.Label(root, borderwidth=1, width=75)
    pathlabel.config(font=('Arial', 7), relief=GROOVE)
    backg.create_window(250, 135, window=pathlabel)

    # Creates button for browsing where to store the model
    browseButton_model = tkk.Button(text="Export To", command=lambda: [exportpath()], width=20)
    backg.create_window(335, 170, window=browseButton_model)

    # Creating label which displays the selected directory
    labelm = tkk.Label(root, text='Select Directory', width=20)
    labelm.config(font=('Arial', 10))
    backg.create_window(150, 170, window=labelm)
    exportp = tk.Label(root, borderwidth=1, width=75)
    exportp.config(font=('Arial', 7), relief=GROOVE)
    backg.create_window(250, 205, window=exportp)

    # Creates a button for resetting all the input fields
    restbutton = tkk.Button(root, text="Reset", command=lambda: [resets()], width=20)
    backg.create_window(170, 400, window=restbutton)

    # Define the progress bar
    bar = tkk.Progressbar(root, length=450)

    backg.create_window(250, 440, window=bar)

    # Button for starting the model building
    loadsbutton = tkk.Button(root, text="Run", command=lambda: [loadmo()], width=20)
    backg.create_window(335, 400, window=loadsbutton)

    # Create a new thread for running the model builder
    th = threading.Thread(target=start_model_builder,
                          args=[root, model_label_entry, entry2, spin, libscale4, pathlabel, exportp])
    root.mainloop()
    # Wait for the thread to finish
    th.join()


# %% Start of Model Builder
def model_builder(CSV_PATH, LABEL_NAME, PERCENTILE, MODEL_NAME, WORKING_DIR, folds, root):
    APP_NAME = "Model Builder"
    spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
    # Read in data from CSV file
    data = spark.read.csv(CSV_PATH, header='true', inferSchema='true')
    # Shuffle data to avoid any bias
    data = data.orderBy(rand())

    # %% Feature Selection
    # Read in data
    dataFeature = pd.read_csv(CSV_PATH)
    # Select all features
    X = dataFeature.drop(columns=[LABEL_NAME])
    # Select the label/target
    Y = dataFeature.loc[:, LABEL_NAME]

    # F_regression function:((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) * std(y)).
    # Select PERCENTILE best features
    selector = SelectPercentile(score_func=f_regression, percentile=PERCENTILE)
    selector.fit(X, Y)

    # show which columns are selected,only indices can be shown
    X_selected = selector.get_support(indices=False)

    feature_list = []
    rejected_list = []
    index = -1
    longestStringLength = 0
    # Places all accepted features into feature_list and all rejected features into rejected_list
    for col in data.columns:
        index = index + 1
        if col == LABEL_NAME:
            index = index - 1
            continue
        elif X_selected[index]:
            feature_list.append(col)
            if len(col) > longestStringLength:
                longestStringLength = len(col)
        else:
            rejected_list.append(col)

    # %% Finding Effective Ranges for accepted features
    # Defines the size of the block of data from a feature which is used for finding best range
    size = round(float(dataFeature.shape[0]) * 0.01)

    # Will contain the effective ranges of each accepted feature
    ranges = []
    for x in range(len(X.columns)):
        if X.columns[x] in feature_list:
            pairs = []
            # Will contain the start and end of the most effective range
            bestSubArray = [0, size]
            # If the feature only contains two different values then a single value will be given instead of a range
            if len(X[X.columns[x]].unique()) == 2:
                op1Sum = 0
                op1Counter = 0
                op2Sum = 0
                op2Counter = 0
                # Goes through all the values in a feature and finds the sum of
                # label value and number of times the two values appear
                for y in range(len(dataFeature)):
                    if X.iloc[y][X.columns[x]] == X[X.columns[x]].unique()[0]:
                        op1Sum = op1Sum + Y.iloc[y]
                        op1Counter = op1Counter + 1
                    else:
                        op2Sum = op2Sum + Y.iloc[y]
                        op2Counter = op2Counter + 1
                # Gets the avg label value for each value in the feature
                op1Sum = op1Sum / op1Counter
                op2Sum = op2Sum / op2Counter
                # Selects which option for the feature is best
                if op1Sum > op2Sum:
                    ranges.append([X[X.columns[x]].unique()[0], X[X.columns[x]].unique()[0]])
                else:
                    ranges.append([X[X.columns[x]].unique()[1], X[X.columns[x]].unique()[1]])
            else:
                # Creates an array containing all the values of the feature paired with their respective label values
                for y in range(len(dataFeature)):
                    pairs.append([X.iloc[y][X.columns[x]], Y.iloc[y]])
                # Sort the pairs based on the value of the feature
                pairs.sort()

                max_current = 0
                index = 0
                # Finds the sum of the label for the first size amount of feature values
                while index < size:
                    max_current = max_current + pairs[index][1]
                    index = index + 1
                # Put the best feature range to be the first legal range
                max_so_far = max_current
                lower = 0
                upper = size
                # For loop checking all other legal ranges and updating the current
                # best range if a region with higher label sum is found
                for z in range(size, len(pairs) - 1):
                    max_current = max_current - pairs[lower][1]
                    max_current = max_current + pairs[upper][1]
                    lower = lower + 1
                    upper = upper + 1
                    if max_so_far < max_current:
                        bestSubArray[0] = pairs[lower][0]
                        bestSubArray[1] = pairs[upper][0]
                        max_so_far = max_current
                ranges.append([bestSubArray[0], bestSubArray[1]])

    # %% Model Building
    # Creates assembler used by all the models
    assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    # Create a training and testing data set
    dataSplit = data.randomSplit([0.8, 0.2])
    train_df = dataSplit[0]
    test_df = dataSplit[1]

    # %% Random Forest

    # specify features and target in random forest
    rf = RandomForestRegressor(labelCol=LABEL_NAME, featuresCol="features")
    # put assemebler and random forest into the pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # hyper-parameter grid
    # evaluate each combination of values (tree size, depth, etc) to find the best
    # also perform k-fold cross validation

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in np.linspace(start=10, stop=20, num=3)]) \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start=5, stop=20, num=3)]) \
        .build()

    # Create cross validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(),
                              numFolds=int(folds))
    # Train rf model with crossvalidator
    cvModel = crossval.fit(train_df)

    # choose the best parameters evalated in hyper-parameter grid
    bestPipeline = cvModel.bestModel
    # Select the best model
    bestModelRF = bestPipeline.stages[1]
    # Get the feature importances for the model
    importances = bestModelRF.featureImportances
    # Create data frame containing the feature importance for the model
    feat_importance = pd.DataFrame(columns=['Feature', 'Score'])
    feat_importance['Feature'] = feature_list
    feat_importance['Score'] = importances
    # Create dataframe containing all predictions on the test data set
    prediction = bestModelRF.transform(assembler.transform(test_df))
    resultRF = prediction.toPandas()

    # %% Linear Regression

    lr = LinReg(labelCol=LABEL_NAME, featuresCol='features')

    # Create the ParamGrid for Cross Validation
    lrParamGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.maxIter, [1, 5, 10]) \
        .build()
    # Create the evaluator
    lrEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol=LABEL_NAME, metricName="rmse")

    # Create the cross validator
    lrCrossValidation = CrossValidator(estimator=lr,
                                       estimatorParamMaps=lrParamGrid,
                                       evaluator=lrEvaluator,
                                       numFolds=int(folds))
    # Train the model with the cross validator and select the best model
    lrBestModel = lrCrossValidation.fit(assembler.transform(train_df))
    bestModelLR = lrBestModel.bestModel
    # Use the model on the training data and put the results in a dataframe
    dataLR = assembler.transform(test_df)
    lr_predictions = bestModelLR.transform(dataLR)
    resultLI = lr_predictions.toPandas()
    # %% Decision Tree Regressor

    # Create the decision tree model
    dt = DecisionTreeRegressor(labelCol=LABEL_NAME, featuresCol='features')

    # Create the ParamGrid for Cross Validation
    dtParamGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [2, 5, 10]) \
        .addGrid(dt.maxBins, [10, 20, 40]) \
        .build()
    # Create the evaluator
    dtEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol=LABEL_NAME, metricName="rmse")

    # Create the cross validator
    dtCrossValidation = CrossValidator(estimator=dt,
                                       estimatorParamMaps=dtParamGrid,
                                       evaluator=dtEvaluator,
                                       numFolds=int(folds))
    # Perform the cross validation
    dtBestModel = dtCrossValidation.fit(assembler.transform(train_df))

    # choose the best parameters evalated in hyper-parameter grid
    bestModelDT = dtBestModel.bestModel
    # Get the feature importances for the model
    importancesDT = bestModelDT.featureImportances
    # Create a dataframe containing the feature importances
    feat_importanceDT = pd.DataFrame(columns=['Feature', 'Score'])
    feat_importanceDT['Feature'] = feature_list
    feat_importanceDT['Score'] = importancesDT

    # Use the model to create a dataframe containing the predictions for the test data set
    dataDT = assembler.transform(test_df)
    dt_predictions = bestModelDT.transform(dataDT)
    resultDT = dt_predictions.toPandas()
    # %% Model Comparison
    # Stores the total different between the predicted values of a model vs the actual value
    weights = [0, 0, 0]
    pandasTest = test_df.toPandas()
    # Finds the sum of the relationship between the predicted value of a model and the actual label value
    for x in range(len(resultRF)):
        weights[0] += (float(resultRF.iloc[x]["prediction"]) / float(pandasTest.iloc[x][LABEL_NAME]))
        weights[1] += (float(resultLI.iloc[x]["prediction"]) / float(pandasTest.iloc[x][LABEL_NAME]))
        weights[2] += (float(resultDT.iloc[x]["prediction"]) / float(pandasTest.iloc[x][LABEL_NAME]))
    # Gets the average
    weights[0] = weights[0] / x
    weights[1] = weights[1] / x
    weights[2] = weights[2] / x

    # Turn the weights positive so they can more easily be compared
    absVal = [0, 0, 0]
    absVal[0] = abs(1 - weights[0])
    absVal[1] = abs(1 - weights[1])
    absVal[2] = abs(1 - weights[2])
    # Variable to store the index of the best model
    bestModel = -1
    bestModelValue = 1000
    # Finds out which model is the best
    for x in range(len(absVal)):
        if absVal[x] < bestModelValue:
            bestModelValue = absVal[x]
            bestModel = x

    # Changes the working directory to the provided saving location
    os.chdir(r"" + WORKING_DIR)
    # Checks if the provided model name already has a folder creates the folder if it does not exist
    isdir = os.path.isdir(MODEL_NAME)
    if not isdir:
        os.mkdir(MODEL_NAME)
    os.chdir(MODEL_NAME)

    # Saves the best model
    if bestModel == 0:
        isList = True
        featureImportance = feat_importance
        modelType = "RF"
        assembler.save(os.path.join(WORKING_DIR + "/" + MODEL_NAME, 'assembler'))
        bestModelRF.save(os.path.join(WORKING_DIR + "/" + MODEL_NAME, "model"))
    elif bestModel == 1:
        assembler.save(os.path.join(WORKING_DIR + "/" + MODEL_NAME, 'assembler'))
        bestModelLR.save(os.path.join(WORKING_DIR + "/" + MODEL_NAME, "model"))
        isList = False
        modelType = "LR"
    elif bestModel == 2:
        isList = True
        featureImportance = feat_importanceDT
        assembler.save(os.path.join(WORKING_DIR + "/" + MODEL_NAME, 'assembler'))
        bestModelDT.save(os.path.join(WORKING_DIR + "/" + MODEL_NAME, "model"))
        modelType = "DT"

    # %% Report Generation

    # Creates the PDF file and creates a new page
    pdf = FPDF()
    pdf.add_page()
    # Sets the font
    pdf.set_font('Times', 'B', 16.0)
    # Prints the model type to the PDF
    pdf.cell(0, 10, "Model Type", 0, 1, "C")
    pdf.set_font('Courier', '', 13.0)
    pdf.cell(0, 10, modelType, 0, 1, "C")
    # Prints the feature importance to the PDF
    pdf.set_font('Times', 'B', 16.0)
    pdf.cell(0, 10, "Feature Importances", 0, 1, "C")
    # Some models do not have feature importance in which case none is printed
    if not isList:
        pdf.set_font('Courier', '', 13.0)
        pdf.cell(0, 10, "None", 0, 1, "C")
    else:
        bestFeature = 'placeholder'
        bestFeatVal = -1
        prevBestVal = 2
        ind = 0
        # Goes through all the feature importances and places them in ranked order based on importance
        for y in range(len(featureImportance)):
            # Finds the feature with the highest feature importance which has not yet been printed
            for x in range(len(featureImportance)):
                if bestFeatVal < featureImportance.iloc[x]['Score'] < prevBestVal:
                    bestFeature = featureImportance.iloc[x]['Feature']
                    bestFeatVal = featureImportance.iloc[x]['Score']
                    ind = x
            prevBestVal = bestFeatVal
            position = y + 1
            pdf.set_font('Courier', '', 13.0)
            # Finds number of spaces to print between feature name and feature importance value for formatting
            numberOfSpaces = (longestStringLength - len(featureImportance.iloc[ind]['Feature']))
            spaces = " " * (numberOfSpaces + 1)
            tempString = str(position) + ": " + bestFeature + " " + spaces + str(round(bestFeatVal, 5))
            pdf.cell(0, 10, tempString, 0, 1, "C")
            bestFeatVal = -1
    # Prints the feature breakdown to the PDF
    pdf.set_font('Times', 'B', 16.0)
    pdf.cell(0, 10, "Feature Breakdown", 0, 1, "C")

    index = 0
    # Goes through all the features in the dataset
    for col in data.columns:
        # Ignores the label column
        if col == LABEL_NAME:
            continue
        # If the current feature was rejected there is no information given about it
        elif col in rejected_list:
            pdf.set_font('Courier', '', 13.0)
            pdf.cell(0, 10, col + ": Removed", 0, 1, "C")
        # One of the accepted features
        else:
            pdf.set_font('Courier', '', 13.0)
            # If the feature only has 2 possible values only a single value is given as a suggestion and not a range
            if ranges[index][0] == ranges[index][1]:
                pdf.cell(0, 10, col + ": " + "Best Value: " + str(ranges[index][0]), 0, 1, "C")
            else:
                pdf.cell(0, 10,
                         col + ": " + "Effective Range: " + str(ranges[index][0]) + " - " + str(ranges[index][1]), 0,
                         1, "C")
            # Create a new figure
            plt.figure(index)
            plt.clf()
            # Create a scatter plot with the feature values and the corresponding label values
            plt.scatter(dataFeature.loc[:, col], dataFeature.loc[:, LABEL_NAME])
            plt.ticklabel_format(style='plain')
            plt.title(col + " vs " + LABEL_NAME)
            plt.xlabel(col)
            plt.ylabel(LABEL_NAME)
            # Save the figure as a png file to the working dir
            plt.figure(index).savefig("plotImageModelBuilder" + str(index) + ".png")
            # If the y co-ordinate is too high an error will happen where the provided co-ordinate for the x-axis
            # is ignored hence if it gets below this threshold a new page is started
            if pdf.get_y() >= 230:
                pdf.add_page()
            # Places the graph in the PDF
            pdf.image("plotImageModelBuilder" + str(index) + ".png", x=66, w=80, h=60)
            # Deletes the saved image as it has now been added to the PDF
            os.remove("plotImageModelBuilder" + str(index) + ".png")
            index = index + 1
    # Writes the accuracy of the model to the PDF
    pdf.set_font('Times', 'B', 16.0)
    pdf.cell(0, 10, "Accuracy Of Model", 0, 1, "C")
    pdf.set_font('Courier', '', 13.0)
    # If the model over predicted on average
    if (weights[bestModel] - 1) >= 0:
        pdf.cell(0, 10,
                 "The model over-predicts on average by " + str(round(((weights[bestModel] - 1) * 100), 1)) + "%", 0,
                 1, "C")
    # If the model under-predicted on average
    else:
        pdf.cell(0, 10,
                 "The model under-predicts on average by " + str(round(((1 - weights[bestModel]) * 100), 1)) + "%", 0,
                 1, "C")
    # Save the pdf file
    pdf.output('information.pdf', 'F')
    # Quit the UI
    root.quit()


# %% Start of Main
# Start the main loop
if __name__ == '__main__':
    main()

gc.collect()
