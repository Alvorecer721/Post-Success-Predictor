# %% Imports
import PyPDF2
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import os
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.regression import DecisionTreeRegressionModel
import tkinter as tk
import tkinter.ttk as tkk
from ttkthemes import ThemedStyle
from tkinter import *
from tkinter import filedialog, GROOVE
import pandas as pd
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
    # Gets the user to browse for the CSV file
    def getCSV():
        csv_file_path = filedialog.askopenfilename()
        pathlabel.config(text=csv_file_path)

    # Gets the user to browse for the model dir
    def exportpath():
        model_paths = filedialog.askdirectory()
        exportp.config(text=model_paths)

    # Creates the UI
    root = Tk()
    # Defines size of the UI
    root.geometry("500x315")
    root.title("Predictor")
    style = ThemedStyle(root)
    style.set_theme("scidblue")
    # Creates canvas for items to be placed in
    backg = tk.Canvas(root, width=500, height=400, bg='black', relief='groove')
    backg.pack()
    # Reads in the background image for the UI
    filename = PhotoImage(file=resource_path("src\\bg.png"))
    background_label = Label(root, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Creates and places the hearder
    headlabel = tkk.Label(root, text='Predictor', borderwidth=2, width=50)
    headlabel.config(font=('system', 25, "bold"), anchor=CENTER, relief='ridge')
    backg.create_window(250, 50, window=headlabel)
    # Creates and places the label for the name of the output file
    entry1 = tkk.Entry(root, width=20)
    backg.create_window(335, 240, window=entry1)
    lb1entry = tkk.Label(root, text='Output Name:', width=20, borderwidth=1)
    lb1entry.config(font=('Arial', 10))
    backg.create_window(150, 240, window=lb1entry)

    # Creates the button for browsing for the CSV file
    browseButton_CSV = tkk.Button(root, text="Import CSV", command=getCSV, width=20)
    backg.create_window(335, 100, window=browseButton_CSV, )
    labelCSV = tkk.Label(root, text='Select The CSV File:', width=20, borderwidth=1)
    labelCSV.config(font=('Arial', 10))
    backg.create_window(150, 100, window=labelCSV)
    # Creates the label for showing which CSV file has been selected
    pathlabel = tk.Label(root, borderwidth=1, width=75)
    pathlabel.config(font=('Arial', 7), relief=GROOVE)
    backg.create_window(250, 135, window=pathlabel)

    # Creates and places the button for selecting the model directory
    browseButton_model = tkk.Button(text="Model Directory", command=lambda: [exportpath()], width=20)
    backg.create_window(335, 170, window=browseButton_model)
    labelm = tkk.Label(root, text='Select Model Directory', width=20, borderwidth=1)
    labelm.config(font=('Arial', 10))
    backg.create_window(150, 170, window=labelm)
    # Creates the label for showing which model dir has been selected
    exportp = tk.Label(root, borderwidth=1, width=75)
    exportp.config(font=('Arial', 7), relief=GROOVE)
    backg.create_window(250, 205, window=exportp)
    # Creates the button for starting the prediction once all inputs have been selected
    loadsbutton = tkk.Button(root, text="Run", command=lambda: [prediction(
        pathlabel.cget("text"), exportp.cget("text"), entry1.get(), root)], width=20)
    backg.create_window(335, 280, window=loadsbutton)

    root.mainloop()


# %% Predictor

def prediction(CSV_PATH, WORKING_DIR, OUTPUT_NAME, root):
    spark = SparkSession.builder.appName("Model Predictor").getOrCreate()
    # Read in the data from the given CSV file
    data = pd.read_csv(CSV_PATH)
    # Change the working dir to the given model
    os.chdir(r"" + WORKING_DIR)
    # Opens the information file
    fileObj = open("information.pdf", 'rb')
    # Reads the PDF file
    pdfReader = PyPDF2.PdfFileReader(fileObj)
    # Gets the first page
    pageObj = pdfReader.getPage(0)
    # Extracts what type of model it is from the information document
    modelType = pageObj.extractText()[10:12]
    # Loads the saved assembler
    am = VectorAssembler.load(os.path.join(WORKING_DIR, 'assembler'))
    # Loads in the correct model based on the information file
    md = " "
    if modelType == "RF":
        md = RandomForestRegressionModel.load(os.path.join(WORKING_DIR, 'model'))
    elif modelType == "LR":
        md = LinearRegressionModel.load(os.path.join(WORKING_DIR, 'model'))
    elif modelType == "DT":
        md = DecisionTreeRegressionModel.load(os.path.join(WORKING_DIR, 'model'))
    # Create dataframe from the provided CSV data
    df_sp = spark.createDataFrame(data)
    # Do predictions on the dataframe
    compare = md.transform(am.transform(df_sp))
    result = compare.toPandas()
    # Add a new column to the dataframe with the predictions
    data['prediction'] = result.iloc[:]["prediction"]

    # Check if the outputs folder has already been created if not create it
    path = "outputs"
    isdir = os.path.isdir("outputs")
    if not isdir:
        os.mkdir(path)

    os.chdir(path)
    # Save the CSV file to the outputs folder
    data.to_csv(OUTPUT_NAME + ".csv", index=False)
    # Close the UI
    root.quit()


# %% Start of Main
if __name__ == '__main__':
    main()

gc.collect()
