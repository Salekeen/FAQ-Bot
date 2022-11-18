import re
def get_synonyms(text):
    """
    It replaces the word "sibling" with "brother and sister" and the word "parent" with "father and
    mother"
    
    Args:
      text: The text to be processed.
    
    Returns:
      The text with the synonyms
    """
    text = re.sub(r'[Ss]ibling[s]',r'brother and sister',text)
    text = re.sub(r'[Pp]arent[s]',r'father and mother',text)
    return text