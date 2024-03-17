
import numpy as np


def round_and_string(
    number: np.float64 | float,
    decimals: int = 5,
    integer_floats_to_ints: bool = True,
    large_scientific_limit: float = 1e9,
    small_scientific_limit: float = 1e-4,
):
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
    if decimals < 1:
        decimals = 0

    if np.isfinite(number):  # check the input is not NaN
        decimal = number % 1
        if number == 0:
            if integer_floats_to_ints is True and decimal == 0:
                number = int(number)
            out = number
        elif abs(number) >= large_scientific_limit or abs(number) <= small_scientific_limit:
            if decimal != 0:
                # special formatting is only applied when the number is a float in the ranges specified above
                out = str("{:0." + str(decimals) + "e}").format(number)
                excess_zeros = "".join(["0"] * decimals)
                out = out.replace("." + excess_zeros + "e", "e")
            else:
                out = number
                if integer_floats_to_ints is True and decimal == 0:
                    out = int(out)
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
    elif text_color is None:
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
