import re
def get_synonyms(text):
    text = re.sub(r'[Ss]ibling[s]',r'brother and sister',text)
    text = re.sub(r'[Pp]arent[s]',r'father and mother',text)
    return text