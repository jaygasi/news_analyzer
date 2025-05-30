from langdetect import detect, detect_langs
import re


def clean_string(s):
    if s is None or len(s) == 0:
        return ""
    return s.encode("ascii", "ignore").decode().strip()


def join_items(item_list):
    return ",".join(item_list)


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def language_detection(text, method="single"):
    if (method.lower() != "single"):
      result = detect_langs(text)
    else:
      result = detect(text)

    return result


def clean_text(text):
    text = text.strip()
    if text == "":
        return text

    #  Cut off text after last dot
    index = text.rfind('.')
    output = ''
    if index > 0:
        output = text[0: index]

    #  Check if the last char is a dot
    if output[-1] != '.':
        output += '.'

    return output


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def capitalize_first_word(sentence):
    # First, convert the entire sentence to lowercase
    lower_case_sentence = sentence.lower()

    # Next, use regular expressions to capitalize the first letter of each sentence
    capitalized_sentence = re.sub(r"(?<=\.\s)(\w+)", lambda x: x.group().capitalize(), lower_case_sentence.capitalize())

    return capitalized_sentence
