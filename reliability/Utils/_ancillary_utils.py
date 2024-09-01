from itertools import chain, repeat
from pathlib import Path

import numpy as np


def is_allowed_character(character: str) -> bool:
    """Checks if a character is allowed.

    Args:
    ----
        character (str): The character to check.

    Returns:
    -------
        bool: True if the character is allowed, False otherwise.

    """
    return character in "YN"


def validate_yes_no_prompt(prompt_msg) -> str:
    """Validates a yes/no prompt input and returns the choice.

    Args:
    ----
        prompt_msg (str): The message to display as the prompt.

    Returns:
    -------
        str: The validated choice, either 'Y' or 'N'.

    """
    bad_input_msg = "Invalid choice. Please specify Y or N"
    prompts = chain([prompt_msg], repeat(f"{bad_input_msg}\n{prompt_msg}"))
    replies = map(input, prompts)
    uppercased_replies = map(str.upper, replies)
    stripped_replies = map(str.strip, uppercased_replies)
    choice: str = next(filter(is_allowed_character, stripped_replies))
    return choice


def update_path(path: Path) -> Path:
    """Update the given path by adding '(new)' to the filename if it already exists.

    Args:
    ----
        path (Path): The original path.

    Returns:
    -------
        Path: The updated path with '(new)' added to the filename.

    """
    new_path: Path = path.with_name(f"{path.stem}(new){path.suffix}")
    return new_path


def write_df_to_xlsx(df, path: Path, **kwargs) -> None:
    """Writes a dataframe to an xlsx file
    For use exclusively by the Convert_data module

    Parameters
    ----------
    df : dataframe
        The dataframe to be written
    path : Path
        The file path to the xlsx file.

    Returns
    -------
    None
        Writing the dataframe is the only action from this function.

    Notes
    -----
    The path must include the full file path including the extension. It is also
    necessary to use r at the start to specify raw text. See the
    `documentation <https://reliability.readthedocs.io/en/latest/Converting%20data%20between%20different%20formats.html>`_ for an example.

    """
    # this section checks whether the file exists and reprompts the user based on their choices
    if Path.exists(path):
        colorprint(
            "WARNING: the specified output file already exists",
            text_color="red",
        )
        prompt_msg = "Do you want to overwrite the existing file (Y/N): "
        choice: str = validate_yes_no_prompt(prompt_msg)
        if choice == "N":
            write_df_to_xlsx(df, update_path(path))
            return
    # this section does the writing
    keys = kwargs.keys()
    if "excel_writer" in keys:
        colorprint(
            "WARNING: excel_writer has been overridden by path. Please only use path to specify the file path for the xlsx file to write.",
            text_color="red",
        )
        kwargs.pop("excel_writer")
    write_index = kwargs.pop("index") if "index" in keys else False
    df.to_excel(path, index=write_index, engine="xlsxwriter", **kwargs)


def round_and_string(
    number: np.float64 | float,
    decimals: int | None = 5,
    integer_floats_to_ints: bool = True,
    large_scientific_limit: float = 1e9,
    small_scientific_limit: float = 1e-4,
) -> str:
    """This function is used to round a number to a specified number of decimals and convert to string.
    It is used heavily in the formatting of the parameter titles within reliability.Distributions

    The rules applied are slightly different than rounding to a number of significant figures as it keeps more
    preceeding zeros. Special formatting rules are applied when:
        abs(number) <= small_scientific_limit
        abs(number) >= large_scientific_limit
        number = 0

    Parameters
    ----------
    number : float
        The number to be rounded
    decimals : int
        The number of decimals (not including preceeding zeros) that are to be
        in the output
    integer_floats_to_ints : bool, optional
        Default is True. Removes trailing zeros from floats if there are no
        significant decimals (eg. 12.0 becomes 12).
    large_scientific_limit : int, float, optional
        The limit above which to keep numbers formatted in scientific notation. Default is 1e9.
    small_scientific_limit : int, float, optional
        The limit below which to keep numbers formatted in scientific notation. Default is 1e-4.

    Returns
    -------
    out : string
        The number formatted and converted to string.

    Notes
    -----
    Examples (with decimals = 5):
        original: -1e+20  // new: -1e+20
        original: -100000000.0  // new: -100000000
        original: -10.0  // new: -10
        original: -5.4857485e-05  // new: -5.48575e-05
        original: -2.54875415e-16  // new: -2.54875e-16
        original: 0.0  // new: 0
        original: 0  // new: 0
        original: 2.54875415e-16  // new: 2.54875e-16
        original: 5.4857485e-05  // new: 5.48575e-05
        original: 10.0  // new: 10
        original: 100000000.0  // new: 100000000
        original: 1e+20  // new: 1e+20

    """
    if decimals is not None and decimals < 1:
        decimals = 0
    if decimals == 0:
        decimals = None  # this is needed for the round function to work properly with some numpy types
    if np.isfinite(number):  # check the input is not NaN
        decimal = number % 1
        if number == 0:
            if integer_floats_to_ints is True and decimal == 0:
                number = int(number)
            out = number
        elif abs(number) >= large_scientific_limit or abs(number) <= small_scientific_limit:
            if decimals is not None and decimal != 0:
                # special formatting is only applied when the number is a float in the ranges specified above
                out = str("{:0." + str(decimals) + "e}").format(number)
                excess_zeros = "".join(["0"] * decimals)
                out = out.replace("." + excess_zeros + "e", "e")
            else:
                out = number
        else:
            out = round(number, decimals)
            if integer_floats_to_ints is True and decimal == 0:
                out = int(out)
    else:  # NaN
        out = number
    return str(out)


def colorprint(
    string,
    text_color=None,
    background_color=None,
    bold=False,
    underline=False,
    italic=False,
):
    """Provides easy access to color printing in the console.

    This function is used to print warnings in red text, but it can also do a
    lot more.

    Parameters
    ----------
    string
    text_color : str, None, optional
        Must be either grey, red, green, yellow, blue, pink, or turquoise. Use
        None to leave the color as white. Default is None.
    background_color : str, None, optional
        Must be either grey, red, green, yellow, blue, pink, or turquoise. Use
        None to leave the color as the transparent. Default is None.
    bold : bool, optional
        Default is False.
    underline : bool, optional
        Default is False.
    italic : bool, optional
        Default is False.

    Returns
    -------
    None
        The output is printed to the console.

    Notes
    -----
    Some flexibility in color names is allowed. eg. red and r will both give red.

    As there is only one string argument, if you have multiple strings to print,
    you must first combine them using str(string_1,string_2,...).

    """
    text_colors = {
        "grey": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "pink": "\033[95m",
        "turquoise": "\033[96m",
        None: "\033[39m",
    }

    background_colors = {
        "grey": "\033[100m",
        "red": "\033[101m",
        "green": "\033[102m",
        "yellow": "\033[103m",
        "blue": "\033[104m",
        "pink": "\033[105m",
        "turquoise": "\033[106m",
        None: "\033[49m",
    }

    BOLD = "\x1b[1m" if bold is True else "\x1b[21m"

    UNDERLINE = "\x1b[4m" if underline is True else "\x1b[24m"

    ITALIC = "\x1b[3m" if italic is True else "\x1b[23m"

    if type(text_color) not in [str, np.str_, type(None)]:
        raise ValueError("text_color must be a string")
    if text_color is None:
        pass
    elif text_color.upper() in ["GREY", "GRAY", "GR"]:
        text_color = "grey"
    elif text_color.upper() in ["RED", "R"]:
        text_color = "red"
    elif text_color.upper() in ["GREEN", "G"]:
        text_color = "green"
    elif text_color.upper() in ["YELLOW", "Y"]:
        text_color = "yellow"
    elif text_color.upper() in ["BLUE", "B", "DARKBLUE", "DARK BLUE"]:
        text_color = "blue"
    elif text_color.upper() in ["PINK", "P", "PURPLE"]:
        text_color = "pink"
    elif text_color.upper() in [
        "TURQUOISE",
        "TURQ",
        "T",
        "CYAN",
        "C",
        "LIGHTBLUE",
        "LIGHT BLUE",
        "LB",
    ]:
        text_color = "turquoise"
    else:
        raise ValueError("Unknown text_color. Options are grey, red, green, yellow, blue, pink, turquoise.")

    if type(background_color) not in [str, np.str_, type(None)]:
        raise ValueError("background_color must be a string")
    if background_color is None:
        pass
    elif background_color.upper() in ["GREY", "GRAY", "GR"]:
        background_color = "grey"
    elif background_color.upper() in ["RED", "R"]:
        background_color = "red"
    elif background_color.upper() in ["GREEN", "G"]:
        background_color = "green"
    elif background_color.upper() in ["YELLOW", "Y"]:
        background_color = "yellow"
    elif background_color.upper() in ["BLUE", "B", "DARKBLUE", "DARK BLUE"]:
        background_color = "blue"
    elif background_color.upper() in ["PINK", "P", "PURPLE"]:
        background_color = "pink"
    elif background_color.upper() in [
        "TURQUOISE",
        "TURQ",
        "T",
        "CYAN",
        "C",
        "LIGHTBLUE",
        "LIGHT BLUE",
        "LB",
    ]:
        background_color = "turquoise"
    else:
        raise ValueError("Unknown text_color. Options are grey, red, green, yellow, blue, pink, turquoise.")

    print(
        BOLD + ITALIC + UNDERLINE + background_colors[background_color] + text_colors[text_color] + string + "\033[0m",
    )


def print_dual_stress_fit_results(
    dist_type: str,
    optimizer: str,
    len_failures: int,
    len_right_cens: int,
    CI: float,
    results,
    change_of_parameters,
    shape_change_exceeded,
    shape_change_threshold,
    goodness_of_fit,
    use_level_stress,
    mean_life,
) -> None:
    n: int = len_failures + len_right_cens
    CI_rounded = CI * 100
    if CI_rounded % 1 == 0:
        CI_rounded = int(CI * 100)
    frac_censored = len_right_cens / n * 100
    if frac_censored % 1 < 1e-10:
        frac_censored = int(frac_censored)
    colorprint(
        str("Results from " + dist_type + " (" + str(CI_rounded) + "% CI):"),
        bold=True,
        underline=True,
    )
    print("Analysis method: Maximum Likelihood Estimation (MLE)")
    if optimizer is not None:
        print("Optimizer:", optimizer)
    print(
        "Failures / Right censored:",
        str(str(len_failures) + "/" + str(len_right_cens)),
        str("(" + round_and_string(frac_censored) + "% right censored)"),
        "\n",
    )
    print(results.to_string(index=False), "\n")
    print(change_of_parameters.to_string(index=False))
    if shape_change_exceeded is True:
        print(
            str(
                "The sigma parameter has been found to change significantly (>"
                + str(int(shape_change_threshold * 100))
                + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
            ),
        )
    print("\n", goodness_of_fit.to_string(index=False), "\n")

    if use_level_stress is not None:
        print(
            str(
                "At the use level stress of "
                + round_and_string(use_level_stress[0])
                + ", "
                + round_and_string(use_level_stress[1])
                + ", the mean life is "
                + str(round(mean_life, 5))
                + "\n",
            ),
        )
