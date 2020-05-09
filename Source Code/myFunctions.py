def findAndReplace(df, new_df, oldStr, newStr, row, old_col, new_col):
    """
    This method checks whether oldStr content is in the element (row, old_col) in original dataframe
    If it is, then place newStr content in element (row, new_col) in the destination dataframe
    :param df: original dataframe to be referenced
    :param new_df: destination dataframe where all modifications are performed
    :param oldStr: old string which should be found and replaced
    :param newStr: new string which should replace the old string
    :param row: specific row of dataframe to do find and replacement
    :param old_col: specific column in the original dataframe to find the old string
    :param new_col: specific column in the destination dataframe to place the new string
    """
    if oldStr in str(df.iat[row, old_col]):
        new_df.iat[row, new_col] = newStr


def combineMultipleElem(df, start_col, end_char, target_col):
    """
    This method combines content in multiple consistent elements into a single element in the same row,
    and separate by ','
    :param df: dataframe where multiple elements are combined
    :param start_col: index of column inclusive where combination starts
    :param end_char: if character appear in a column, that column is where the combination ends
    :param target_col: index of column where combined content should be place in
    """
    for row in range(len(df)):
        temp_col = start_col
        topicTags = ''

        while end_char not in str(df.iat[row, temp_col]):
            topicTags += str(df.iat[row, temp_col]) + ', '
            temp_col += 1
            if end_char in str(df.iat[row, temp_col]):
                topicTags += str(df.iat[row, temp_col])
                df.iat[row, target_col] = topicTags


def shiftElementPosMult(df, start_col, start_str):
    """
    This methods checks which column of element contains start_str in its element to the right of start column
    and force multiple elements on the right side to fill the gap
    :param df: dataframe where this method should be performed
    :param start_col: index of column where checks of whether start_str presents in the element starts
    :param start_str: the string which are to be checked
    """
    for row in range(len(df)):
        temp_col = start_col
        count = 0

        while start_str not in str(df.iat[row, temp_col]):
            temp_col += 1
            count += 1

        for copy in range(temp_col, len(df.columns) - 1):
            df.iat[row, copy - count] = df.iat[row, copy]


def shiftElementPosSingle(df, row, col, s):
    """
    This methods checks if content of s is in the element (row, col)
    If it is, shift all the elements in the right side left by 1 position to overwrite this element
    :param df: dataframe where this function should be performed
    :param row: row of the element
    :param col: col of the element
    :param s: string to be checked which it's contained in the element
    """
    if s in str(df.iat[row, col]):
        for copy in range(col+1, len(df.columns)-1):
            df.iat[row, copy - 1] = df.iat[row, copy]


def cleanData(temp, str1, str2):
    """
    This method is used to remove garbled character on the left and right side of the actual content through out
    entire dataframe
    :param temp: the dataframe to perform data cleaning
    :param str1: garbled character to be removed on the left side of actual contents, this should be done first
    :param str2: garbled character to be removed on both side of actual contents
    """
    for column in temp.columns:
        temp[column] = temp[column].str.lstrip(str1)
        temp[column] = temp[column].str.strip(str2)


def calculateRatio(x, y):
    """
    This function is used to calculate ratio between x and y
    :param x: input x
    :param y: input y
    :return: if input y is 0 then just return 0, otherwise return x / y
    """
    return 0 if y == 0 else x / y
