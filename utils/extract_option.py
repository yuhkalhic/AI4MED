import re


def extract_option(pred):
    # 1. get A/B/C/D
    for pattern in [
        r'<answer>(.*?)</answer>',
        r'^([A-Z])[.,:]',
        r'Answer:\s*([A-Z])\s*',
    ]:
        match = re.search(pattern, pred, re.DOTALL)
        if match is not None:
            pred = match.group(1)

    # 2. remove <>
    pred = pred.replace("<", "").replace(">", "")
    pred = pred.strip()
    return pred