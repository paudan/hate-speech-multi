# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import string
import re
import warnings
import ftfy
import emoji
import datefinder
from dateutil.parser import UnknownTimezoneWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UnknownTimezoneWarning)
tqdm.pandas()


def preprocess_text(text):
    # First, use ftfy to fix any encoding issues
    if hasattr(text, '__len__'):
        text = ftfy.fix_text(text)

        # Remove dates
        text = remove_dates(text)
        
        # Custom rules for punctuation fixing
        rules = [
            # Remove http and https links
            (r'https?://\S+', ''),
            # Remove mentions
            (r'@\S+', ''),
            # Remove tags
            (r'#\S+', ''),
            # Remove IPs
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ''),
            # Remove NEWLINE_TOKEN
            ('NEWLINE_TOKEN', ''),
            # Remove consecutive repetitive punctuation, but keep a maximum of two for emphasis (e.g., !!)
            (r'([,\.?!])\1{2,}', r'\1\1'),
            # Add space after comma, period, question mark, or exclamation mark if not followed by space
            (r'([,\.?!])(?=[^\s])', r'\1 '),
            # Remove space before comma, period, question mark, or exclamation mark
            (r'\s+([,\.?!])', r'\1'),
            # Fix multiple spaces
            (r'\s{2,}', ' '),
            # Ensure numbers have space before and after, except when punctuation or hyphen follows
            (r'(\d)(?=[^\s\d,\.?!-])', r'\1 '),
            (r'(?<=[^\s\d-])(\d)', r' \1')
        ]
    
        # Replace emoji with :shortcode:
        text = emoji.demojize(text, delimiters=(" ::", ":: "))
        # text = emoji.replace_emoji(text)
    
        # Apply each rule
        for pattern, replacement in rules:
            text = re.sub(pattern, replacement, text)

        stripped = string.punctuation + string.whitespace    
        text = text.strip(stripped)
    else:
        text = ''
    return text

def remove_dates(text):
    matches = datefinder.find_dates(text, source=True)
    replaced = []
    for match in matches:
        replaced.append(match[1])
    for replacement in replaced:
        text = text.replace(replacement, '')
    return text
