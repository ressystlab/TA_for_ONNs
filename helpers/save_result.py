import pandas as pd

def save_dict_to_excel(data_dict, filename="output.xlsx", orientation="columns", sheet_name="Sheet1"):
    """
    Save a dictionary of lists to an Excel file.

    Parameters:
    - data_dict (dict): Dictionary of lists to write.
    - filename (str): Path to the Excel file to save.
    - orientation (str): "columns" (default) to write keys as column headers, or "rows" to write keys as row labels.
    - sheet_name (str): Name of the sheet in the Excel file.
    """
    if orientation not in {"columns", "rows"}:
        raise ValueError("orientation must be either 'columns' or 'rows'")

    # Convert the dictionary to a DataFrame
    if orientation == "columns":
        df = pd.DataFrame(data_dict)
    else:  # "rows"
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.columns = [f"Value {i+1}" for i in range(df.shape[1])]

    # Save to Excel
    df.to_excel(filename, index=(orientation == "rows"), sheet_name=sheet_name)
    print(f"Saved to {filename} (orientation: {orientation})")