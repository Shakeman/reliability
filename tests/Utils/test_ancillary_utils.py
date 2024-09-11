from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from reliability.Utils._ancillary_utils import (
    colorprint,
    is_allowed_character,
    round_and_string,
    update_path,
    validate_yes_no_prompt,
    write_df_to_xlsx,
)


def test_is_allowed_character():
    assert is_allowed_character("Y") is True
    assert is_allowed_character("N") is True
    assert is_allowed_character("A") is False
    assert is_allowed_character("B") is False
    assert is_allowed_character("Z") is False


def test_validate_yes_no_prompt(monkeypatch: pytest.MonkeyPatch):
    # Test valid input 'Y'
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    assert validate_yes_no_prompt("Are you sure?") == "Y"

    # Test valid input 'N'
    monkeypatch.setattr("builtins.input", lambda _: "N")
    assert validate_yes_no_prompt("Are you sure?") == "N"

    # Test invalid input 'A' followed by valid input 'Y'
    responses: Iterator[str] = iter(["A", "Y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    assert validate_yes_no_prompt("Are you sure?") == "Y"

    # Test invalid input 'B' followed by valid input 'N'
    responses: Iterator[str] = iter(["B", "N"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    assert validate_yes_no_prompt("Are you sure?") == "N"


def test_update_path():
    # Test case: (new) added to file name
    path = Path("C:\\Users\\Current User\\Desktop\\XCN.xlsx")
    expected_result = Path("C:\\Users\\Current User\\Desktop\\XCN(new).xlsx")
    assert update_path(path) == expected_result


def test_round_and_string():
    # Test case: number = 0
    assert round_and_string(0) == "0"

    # Test case: number = 10.0
    assert round_and_string(10.0) == "10.0"

    # Test case: number = -5.4857485e-05
    assert round_and_string(-5.4857485e-05) == "-5.48575e-05"

    # Test case: number = 2.54875415e-16
    assert round_and_string(2.54875415e-16) == "2.54875e-16"

    # Test case: number = -1e+20
    assert round_and_string(-1e20) == "-1e+20"

    # Test case: number = 1e+20
    assert round_and_string(1e20) == "1e+20"

    # Test case: number = -2.54875415e-16
    assert round_and_string(-2.54875415e-16) == "-2.54875e-16"

    # Test case: number = 5.4857485e-05
    assert round_and_string(5.4857485e-05) == "5.48575e-05"

    # Test case: number = -10.0
    assert round_and_string(-10.0) == "-10.0"

    # Test case: number = -5.4857485e-05, decimals = 3
    assert round_and_string(-5.4857485e-05, decimals=3) == "-5.486e-05"

    # Test case: number = 10.0, decimals = 3
    assert round_and_string(10.0, decimals=3) == "10.0"

    # Test case: number = -2.54875415e-16, decimals = 3
    assert round_and_string(-2.54875415e-16, decimals=3) == "-2.549e-16"

    # Test case: number = 5.4857485e-05, decimals = 3
    assert round_and_string(5.4857485e-05, decimals=3) == "5.486e-05"

    # Test case: number = -1e+20, decimals = 3
    assert round_and_string(-1e20, decimals=3) == "-1e+20"

    # Test case: number = 1e+20, decimals = 3
    assert round_and_string(1e20, decimals=3) == "1e+20"

    # Test case: number = 0.123456789, decimals = 8
    assert round_and_string(0.123456789, decimals=8) == "0.12345679"

    # Test case: number = 0.123456789, decimals = 0
    assert round_and_string(0.123456789, decimals=0) == "0"

    # Test case: number = np.nan
    assert round_and_string(np.nan) == "nan"

    # Test case: number = np.inf
    assert round_and_string(np.inf) == "inf"

    # Test case: number = -np.inf
    assert round_and_string(-np.inf) == "-inf"


def test_colorprint():
    # Test case: Print string in red color
    colorprint("Hello, world!", text_color="red")

    # Test case: Print string in green color with bold and underline
    colorprint("Hello, world!", text_color="green", bold=True, underline=True)

    # Test case: Print string with yellow background color
    colorprint("Hello, world!", background_color="yellow")

    # Test case: Print string with blue text color and pink background color
    colorprint("Hello, world!", text_color="blue", background_color="pink")

    # Test case: Print string with invalid text color
    with pytest.raises(ValueError) as e:
        colorprint("Hello, world!", text_color="invalid_color")
    assert str(e.value) == "Unknown text_color. Options are grey, red, green, yellow, blue, pink, turquoise."

    # Test case: Print string with invalid background color
    with pytest.raises(ValueError) as e:
        colorprint("Hello, world!", background_color="invalid_color")
    assert str(e.value) == "Unknown text_color. Options are grey, red, green, yellow, blue, pink, turquoise."


def test_write_df_to_xlsx(monkeypatch: pytest.MonkeyPatch):
    current_path: Path = Path.cwd()
    folder_path: Path = current_path / "tests"
    folder_path: Path = folder_path / "_excel_files"
    path: Path = folder_path / "test.xlsx"
    new_path: Path = folder_path / "test(new).xlsx"
    generic_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Test case: Writing dataframe to xlsx file
    write_df_to_xlsx(generic_df, path)
    assert Path.exists(path)

    # Test case: Writing dataframe to existing xlsx file and choosing not to overwrite
    monkeypatch.setattr("builtins.input", lambda _: "N")
    write_df_to_xlsx(generic_df, path)
    assert Path.exists(new_path)

    # Test case: Writing dataframe to existing xlsx file and choosing to overwrite
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    write_df_to_xlsx(generic_df, path)
    assert Path.exists(path)

    Path(path).unlink()

    # Test case: Writing dataframe to xlsx file with additional kwargs
    kwargs = {"sheet_name": "Sheet1", "header": False}
    write_df_to_xlsx(generic_df, path, **kwargs)
    assert Path.exists(path)

    Path(path).unlink()
    Path(new_path).unlink()
