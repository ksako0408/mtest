import re
import numpy as np
import gzip

_base_path = "/constant/polyMesh/"
_file_open = open
_file_extension = ""
# _base_Path = "/0/polyMesh/"
# _file_open = gzip.open
# _file_extension = ".gz"


def _del_comment(text: str):
    return re.sub(r'/\*[\s\S]*?\*/|//.*', '', text).strip()


def _del_header(text: str):
    return re.sub(r'FoamFile[\s\S]*?\{[\s\S]*?.*\}', '', text).strip()


def read_PointsFile(processor: str):
    file = processor + _base_path + "points" + _file_extension
    with _file_open(file, "r", encoding = "utf-8") as f: text = f.read()
    text = _del_comment(text)
    text = _del_header(text)

    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\)\(', '),(', text)
    text = re.sub(r'\s+', ',', text)
    return eval(text)


def read_FacesFile(processor: str):
    file = processor + _base_path + "faces" + _file_extension
    with _file_open(file, "r", encoding = "utf-8") as f: text = f.read()
    text = _del_comment(text)
    text = _del_header(text)

    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\)[^(]+\(', '),(', text)
    text = re.sub(r'\([0-9]+\(', '((', text)
    text = re.sub(r'\s+', ',', text)
    return eval(text)
    # text = eval(text)
    # return {i: f for i, f in enumerate(text)}


def read_CellFaceFile(processor: str, kind: str):
    file = processor + _base_path + kind + _file_extension
    with _file_open(file, "r", encoding = "utf-8") as f: text = f.read()
    text = _del_comment(text)
    text = _del_header(text)

    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\(\n', '(', text, 1)
    text = re.sub(r'\n\)', ')', text, 1)
    text = re.sub(r'\n', ',', text)
    return eval(text)


def read_CellLevelFile(processor: str):
    file = processor + _base_path + "cellLevel" + _file_extension
    with _file_open(file, "r", encoding = "utf-8") as f: text = f.read()
    text = _del_comment(text)
    text = _del_header(text)

    if '{' in text:
        n = int(text[0:text.find('{')])
        level = int(re.sub(r'.+{', '', re.sub(r'}', '', text)))
        return tuple([level for _ in range(n)])

    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\(\n', '(', text, 1)
    text = re.sub(r'\n\)', ')', text, 1)
    text = re.sub(r'\n', ',', text)
    return eval(text)


def read_BoundaryFile(processor: str):
    file = processor + _base_path + "boundary"
    with open(file, "r", encoding = "utf-8") as f: text = f.read()
    text = _del_comment(text)
    text = _del_header(text)

    text = re.sub(r'[^(]+\(', '}', text, 1)
    text = re.sub(r'\s*\)$', '', text, 1)
    text = re.sub(r'\s*}$', '', text, 1)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\n{', ';', text)
    text = re.sub(r'}\s*', '),dict(name ', text)
    text = re.sub(r'.$', '))', text)
    text = re.sub(r'^\)\,', '(', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r';\)', ')', text)
    text = re.sub(r';', ',', text)
    text = re.sub(r'\s+', '=', text)

    objs = re.findall(r'(?<=name\=)[^,]*(?=,)', text)
    objs = list(set(objs))
    for obj in objs:
        text = text.replace(obj, '"'+obj+'"')

    objs = re.findall(r'(?<=type\=)[^,]*(?=,)', text)
    objs = list(set(objs))
    for obj in objs:
        text = text.replace(obj, '"'+obj+'"')

    objs = re.findall(r'(?<=transform\=)[^,]*(?=,)', text)
    objs = list(set(objs))
    for obj in objs:
        text = text.replace(obj, '"'+obj+'"')

    objs = re.findall(r'(?<=inGroups\=)[^,]*(?=,)', text)
    objs = list(set(objs))
    for obj in objs:
        text = text.replace(obj, "'"+obj+"'")
    return eval(text)


def read_Range_of_Subdomains(points):
    p = np.array(points)
    p_xmin = p[:, 0].min()
    p_ymin = p[:, 1].min()
    p_zmin = p[:, 2].min()
    p_min = (p_xmin, p_ymin, p_zmin)

    p_xmax = p[:, 0].max()
    p_ymax = p[:, 1].max()
    p_zmax = p[:, 2].max()
    p_max = (p_xmax, p_ymax, p_zmax)

    return (p_min, p_max)

