import re

def extract_numbers(input_string: str) -> list:
    """
    Extract all numbers (integers and floats) from a given string.
    :param input_string: The input string containing numbers.
    :return: A list of numbers as floats.
    """
    numbers = re.findall(r'\d+\.\d+|\d+', input_string)
    return [float(num) for num in numbers]