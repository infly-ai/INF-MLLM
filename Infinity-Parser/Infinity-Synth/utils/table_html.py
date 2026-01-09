import random
import pandas as pd
import string


def df_to_custom_html(df):
    """
    Convert DataFrame to HTML table.
    Each column gets a class name (cols1, cols2, ...), useful for CSS styling.
    """
    html = '<table border="1">\n'
    for _, row in df.iterrows():
        html += "  <tr>\n"
        for j, val in enumerate(row):
            class_name = f"cols{j + 1}"  # assign a class to each column
            html += f'    <td class="{class_name}">{val}</td>\n'
        html += "  </tr>\n"
    html += "</table>"
    return html


def get_random_chars_from_string(s: str) -> str:
    """
    Randomly sample characters from a given string.
    Short or long text is chosen by probability.
    """
    if random.random() > 0:
        length = random.randint(4, 8)
    else:
        length = random.randint(25, 45)
    return "".join(random.sample(s, length))


def get_random_chars_from_string_short(s: str) -> str:
    """
    Get short random characters (2–4) from string.
    """
    length = random.randint(2, 4)
    return "".join(random.sample(s, length))


def get_random_float() -> float:
    """
    Generate random float with two decimals.
    """
    return round(random.uniform(-1000, 1000), 2)


def get_random_chars_from_26char() -> str:
    """
    Random sample of 3–8 lowercase English letters.
    """
    letters = string.ascii_lowercase
    length = random.randint(3, 8)
    return "".join(random.sample(letters, length))


def create_random_table(rows: int, cols: int, given_string: str) -> pd.DataFrame:
    """
    Create table data with mixed text, numbers, blanks, and invisible values.
    First half of rows mostly text, second half mixes symbols and numbers.
    Some cells intentionally left blank for table realism.
    """
    table_data = []

    for row_idx in range(rows):
        row_data = []

        # First two columns have structure patterns
        if row_idx < rows / 2:
            row_data.append(get_random_chars_from_string(given_string))  # text
            row_data.append("")  # blank
        else:
            if row_idx % 2 == 1:
                row_data.append("yoy")
                row_data.append(get_random_float() if random.random() > 1.0 else "")
            else:
                row_data.append(get_random_chars_from_string(given_string))
                row_data.append(get_random_float())

        # Fill remaining columns
        for col in range(2, cols):
            if row_idx == 0:  # first row: hidden content in some positions
                invisible_chars = "&nbsp" * random.randint(1, 10)
                row_data.append(invisible_chars)
            else:
                if row_idx < rows / 2 and col < cols / 2:
                    row_data.append("")  # blank zone region
                else:
                    row_data.append("" if random.random() > 0.8 else get_random_float())

        table_data.append(row_data)

    return pd.DataFrame(table_data)


def produce_table_html(given_string: str):
    """
    Generate a table with random rows/columns and converted HTML output.
    Returns (html_string, num_columns)
    """
    rows = random.randint(14, 22)
    cols = random.randint(5, 8)

    table_data = create_random_table(rows, cols, given_string)
    return df_to_custom_html(table_data), cols
