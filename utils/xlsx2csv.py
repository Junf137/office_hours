import os
import pandas as pd


def xlsx_to_csv(xlsx_path: str, out_dir: str, sheet_columns: dict):
    """
    Read an .xlsx file and write each sheet to a separate .csv file.

    Parameters
    ----------
    xlsx_path : str
        Path to the input Excel file.
    out_dir : str, optional
        Directory to save the CSVs in.  Defaults to the same folder as the input file.
    """
    # Make sure pandas can read Excel
    # pip install pandas openpyxl

    # Determine output directory
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(xlsx_path))
    os.makedirs(out_dir, exist_ok=True)

    # Load workbook metadata
    xls = pd.ExcelFile(xlsx_path)
    base_name = os.path.splitext(os.path.basename(xlsx_path))[0]

    # Check for required sheets
    missing_sheets = []
    for sheet_name in sheet_columns:
        if sheet_name not in xls.sheet_names:
            missing_sheets.append(sheet_name)

    if missing_sheets:
        print(f"Error: Missing required sheets in {xlsx_path}:")
        for sheet in missing_sheets:
            print(f"  - {sheet}")

    # Process each required sheet
    for sheet_name, config in sheet_columns.items():
        if sheet_name not in xls.sheet_names:
            continue

        # Read sheet into DataFrame
        df = xls.parse(sheet_name)

        # trim whitespace for columns
        df.columns = df.columns.str.strip()

        # in column, replace Cubical with Cubicle
        df.columns = df.columns.str.replace("Cubical", "Cubicle", regex=False)

        # Check for required columns
        required_columns = config["columns"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Sheet '{sheet_name}' is missing required columns:")
            for col in missing_columns:
                print(f"  - {col}")
            continue

        # strip whitespace from all columns and rows
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        if sheet_name == "Unique Items AppearDisappear":
            # Fix capitalization ensuring "appear" in "disappear" isn't affected
            df["Appear/ Disappear (Change)"] = df["Appear/ Disappear (Change)"].str.replace(r"\bdisappear\b", "Disappear", regex=True)
            df["Appear/ Disappear (Change)"] = df["Appear/ Disappear (Change)"].str.replace(r"\bappear\b", "Appear", regex=True)

            # check this column content, it should only contain "Appear" or "Disappear"
            if not df["Appear/ Disappear (Change)"].isin(["Appear", "Disappear"]).all():
                print(f"Error: Sheet '{sheet_name}' has invalid values in 'Appear/ Disappear (Change)' column.")
                print(f"column values: {df['Appear/ Disappear (Change)'].unique()}")

        # Use only the required columns in the specified order
        df_filtered = df[required_columns]

        # Build output path using the specified output filename
        csv_filename = config["output"]
        csv_path = os.path.join(out_dir, csv_filename)

        # Write CSV
        df_filtered.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":

    sheet_columns = {
        "Location Change": {
            "columns": [
                "Episode",
                "Unique Object",
                "Initial Location Inside Cubicle",
                "Final Location Inside Cubicle",
            ],
            "output": "Object Location Change.csv",
        },
        "Unique Items AppearDisappear": {
            "columns": [
                "Episode",
                "Unique Object",
                "Location inside Cubicle",
                "Appear/ Disappear (Change)",
            ],
            "output": "Object Detection.csv",
        },
        "State Change": {
            "columns": [
                "Episode",
                "Object",
                "Initial State",
                "Final State",
            ],
            "output": "Object State Change.csv",
        },
        "Count Change": {
            "columns": [
                "Episode",
                "Non-unique Object",
                "Initial Count",
                "Final Count",
            ],
            "output": "Object Counting.csv",
        },
    }

    xlsx_to_csv(
        "../csv_files/Alana 2038Q/Local Changes - Alana 2038Q.xlsx",
        "../csv_files/Alana 2038Q",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Alexandria 2008M/Local Changes - Alexandria 2008M.xlsx",
        "../csv_files/Alexandria 2008M",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Amin 2008E/Local Changes - Amin 2008E.xlsx",
        "../csv_files/Amin 2008E",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Daniel 2038Y/Local Changes - Daniel 2038Y.xlsx",
        "../csv_files/Daniel 2038Y",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Emily 2038U/Local Changes - Emily 2038U.xlsx",
        "../csv_files/Emily 2038U",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Fernando 2041S/Local Changes - Fernando 2041S.xlsx",
        "../csv_files/Fernando 2041S",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Jack S. 2038T/Local Changes - Jack S. 2038T.xlsx",
        "../csv_files/Jack S. 2038T",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Jason 2008K/Local Changes - Jason 2008K.xlsx",
        "../csv_files/Jason 2008K",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Josh 2008S/Local Changes - Josh 2008S.xlsx",
        "../csv_files/Josh 2008S",
        sheet_columns,
    )
    xlsx_to_csv(
        "../csv_files/Kristine 2038N/Local Changes - Kristine 2038N.xlsx",
        "../csv_files/Kristine 2038N",
        sheet_columns,
    )
