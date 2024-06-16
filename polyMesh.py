import re
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from operator import itemgetter
import pprint


# ---------------------------------------------------------------------------------------------------------------
# Z+
csys_v = {(-1,  1,  1): 20,  ( 0,  1,  1): 11,  ( 1,  1,  1): 19, \
          (-1,  0,  1): 18,  ( 0,  0,  1):  5,  ( 1,  0,  1): 15, \
          (-1, -1,  1): 21,  ( 0, -1,  1): 12,  ( 1, -1,  1): 22, \
# Z0
          (-1,  1,  0):  8,  ( 0,  1,  0):  2,  ( 1,  1,  0):  7, \
          (-1,  0,  0):  3,                     ( 1,  0,  0):  1, \
          (-1, -1,  0):  9,  ( 0, -1,  0):  4,  ( 1, -1,  0): 10, \
# Z-
          (-1,  1, -1): 24,  ( 0,  1, -1): 14,  ( 1,  1, -1): 23, \
          (-1,  0, -1): 17,  ( 0,  0, -1):  6,  ( 1,  0, -1): 16, \
          (-1, -1, -1): 25,  ( 0, -1, -1): 13,  ( 1, -1, -1): 26}

csys = {v: k for k, v in csys_v.items()}

csys_inv = { 1:  3,  2:  4,  3:  1,  4:  2,  5:  6,  6:  5, \
             7:  9,  8: 10,  9:  7, 10:  8, 11: 13, 12: 14, 13: 11, 14: 12, 15: 17, 16: 18, 17: 15, 18: 16, \
            19: 25, 20: 26, 21: 23, 22: 24, 23: 21, 24: 22, 25: 19, 26: 20}


ref_faces = { 1: { 1,  7, 10, 15, 16, 19, 22, 23, 26}, \
              2: { 2,  7,  8, 11, 14, 19, 20, 23, 24}, \
              3: { 3,  8,  9, 17, 18, 20, 21, 24, 25}, \
              4: { 4,  9, 10, 12, 13, 21, 22, 25, 26}, \
              5: { 5, 11, 12, 15, 18, 19, 20, 21, 22}, \
              6: { 6, 13, 14, 16, 17, 23, 24, 25, 26}}
ref_edges = { 7: { 7, 19, 23}, \
              8: { 8, 20, 24}, \
              9: { 9, 21, 25}, \
             10: {10, 22, 26}, \
             11: {11, 19, 20}, \
             12: {12, 21, 22}, \
             13: {13, 25, 26}, \
             14: {14, 23, 24}, \
             15: {15, 19, 22}, \
             16: {16, 23, 26}, \
             17: {17, 24, 25}, \
             18: {18, 20, 21}}
ref_point = {19: {19}, \
             20: {20}, \
             21: {21}, \
             22: {22}, \
             23: {23}, \
             24: {24}, \
             25: {25}, \
             26: {26}}

# ---------------------------------------------------------------------------------------------------------------
class lattice:
    def __init__(self):
        self.boundaryType = dict()
        self.boundaryName = dict()
        self.neighbourCell = dict()
        self.size = None
        # self.id = None
        self.center = None

    def get_neighbourPoint(self, c, k):
        return tuple(np.array(c) + np.array(csys[k]) * self.size)


class polymesh:
    def __init__(self, proc, base_size, shift, maxCellLevel, lock):
        points = read_PointsFile(proc)
        self.faces = read_FacesFile(proc)
        self.owner = read_CellFaceFile(proc, "owner")
        self.neighbour = read_CellFaceFile(proc, "neighbour")
        self.cellLevel = read_CellLevelFile(proc)
        self.boundaries = read_BoundaryFile(proc)

        self.domainRange = get_Range_of_Subdomains(points)

        lock.acquire()
        try:
            if proc == "processor0":
                for i in range(3):
                    shift[i] = self.domainRange[0][i]
        finally:
            lock.release()
        self.shift = shift[:]
        point_shift = np.array(self.shift)

        lock.acquire()
        try:
            tmp = max(self.cellLevel)
            maxCellLevel.value = max([maxCellLevel.value, tmp])
        finally:
            lock.release()
        self.maxCellLevel = maxCellLevel.value

        self.normalized_Length = (2 ** (self.maxCellLevel + 1)) / base_size
        faces_in_cell = calc_FacesSet_in_Cell(self.owner, self.neighbour)
        self.points_in_cell = calc_PointsSet_in_Cell(self.faces, faces_in_cell)
        self.nrml_points = calc_Normalized_Points(points, self.normalized_Length, point_shift)

        tmp = calc_Normalized_Points(self.domainRange, self.normalized_Length, point_shift)
        self.nrml_domainRange = tuple(tmp.values())

        geom_of_cell = calc_Cell_Geometry(self.points_in_cell, self.nrml_points)
        self.cell_center = {i: v[0] for i, v in geom_of_cell.items()}
        self.cell_size = {i: v[1] for i, v in geom_of_cell.items()}


    def get_ownerCellNo_of_Face(self, face):
        return self.owner[face]

    def calc_FaceSys_of_CellNo(self, cell, face):
        ccenter = self.cell_center[cell]
        fpoints = self.faces[face]
        vset = {csys_v[tuple(np.sign(self.nrml_points[fp] - ccenter))] for fp in fpoints}
        n = int(np.where(np.array(tuple(vset.issubset(ref_face) for ref_face in ref_faces.values())))[0])
        return list(ref_faces.keys())[n]

    def trans_CellNo_to_Cell_Center(self, cell):
        return tuple(self.cell_center[cell])









class readMesh:
    def __init__(self, proc: str, base_size: float, shift: tuple):
        self.processor = proc

        points = read_PointsFile(proc)
        self.faces = read_FacesFile(proc)
        self.owner = read_CellFaceFile(proc, "owner")
        neighbour = read_CellFaceFile(proc, "neighbour")
        cellLevel = read_CellLevelFile(proc)
        self.boundaries = read_BoundaryFile(proc)

        self.max_cellLevel = max(cellLevel)
        self.normalized_length = (2 ** (self.max_cellLevel + 1)) / base_size
        point_shift = np.array(shift)

        faces_in_cell = calc_FacesSet_in_Cell(self.owner, neighbour)
        self.points_in_cell = calc_PointsSet_in_Cell(self.faces, faces_in_cell)
        self.nrml_points = calc_Normalized_Points(points, self.normalized_length, point_shift)
        geom_of_cell = calc_Cell_Geometry(self.points_in_cell, self.nrml_points)
        self.cell_center = {i: v[0] for i, v in geom_of_cell.items()}
        self.cell_size = {i: v[1] for i, v in geom_of_cell.items()}

    def get_ownerCellNo_of_Face(self, face):
        return self.owner[face]

    def calc_FaceSys_of_CellNo(self, cell, face):
        ccenter = self.cell_center[cell]
        fpoints = self.faces[face]
        vset = {csys_v[tuple(np.sign(self.nrml_points[fp] - ccenter))] for fp in fpoints}
        n = int(np.where(np.array(tuple(vset.issubset(ref_face) for ref_face in ref_faces.values())))[0])
        return list(ref_faces.keys())[n]

    def trans_CellNo_to_Cell_Center(self, cell):
        return tuple(self.cell_center[cell])


# ---------------------------------------------------------------------------------------------------------------

def del_comment(text: str):
    return re.sub(r'/\*[\s\S]*?\*/|//.*', '', text).strip()


def del_header(text: str):
    return re.sub(r'FoamFile[\s\S]*?\{[\s\S]*?.*\}', '', text).strip()


def read_CellLevelFile(proc: str):
    file = proc + "/constant/polyMesh/cellLevel"
    with open(file, "r") as f: text = f.read()
    text = del_comment(text)
    text = del_header(text)

    if '{' in text:
        n = int(text[0:text.find('{')])
        level = int(re.sub(r'.+{', '', re.sub(r'}', '', text)))
        return tuple([level for _ in range(n)])

    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\(\n', '(', text, 1)
    text = re.sub(r'\n\)', ')', text, 1)
    text = re.sub(r'\n', ',', text)
    return eval(text)


def read_PointsFile(proc: str):
    file = proc + "/constant/polyMesh/points"
    with open(file, "r") as f: text = f.read()
    text = del_comment(text)
    text = del_header(text)

    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\)\(', '),(', text)
    text = re.sub(r'\s+', ',', text)
    return eval(text)


def read_FacesFile(proc: str):
    file = proc + "/constant/polyMesh/faces"
    with open(file, "r") as f: text = f.read()
    text = del_comment(text)
    text = del_header(text)

    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\)[^(]+\(', '),(', text)
    text = re.sub(r'\([0-9]+\(', '((', text)
    text = re.sub(r'\s+', ',', text)
    text = eval(text)
    return {i: f for i, f in enumerate(text)}


def read_CellFaceFile(proc: str, kind: str):
    file = proc + "/constant/polyMesh/" + kind
    with open(file, "r") as f: text = f.read()
    text = del_comment(text)
    text = del_header(text)

    text = re.sub(r'[^(]+\(', '(', text, 1)
    text = re.sub(r'\(\n', '(', text, 1)
    text = re.sub(r'\n\)', ')', text, 1)
    text = re.sub(r'\n', ',', text)
    return eval(text)



def read_BoundaryFile(proc: str):
    file = proc + "/constant/polyMesh/boundary"
    with open(file, "r") as f: text = f.read()
    text = del_comment(text)
    text = del_header(text)

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

# ---------------------------------------------------------------------------------------------------------------

def calc_FacesSet_in_Cell(owner, neighbour):
    retval = dict()
    for i, c in enumerate(owner):
        if c not in retval:
            retval[c] = {i}
        else:
            retval[c].add(i)
    for i, c in enumerate(neighbour):
        if c not in retval:
            retval[c] = {i}
        else:
            retval[c].add(i)
    return retval


def calc_PointsSet_in_Cell(faces, cells):
    retval = dict()
    for c, fs in cells.items():
        tmp = set()
        for f in fs:
            tmp.update(set(faces[f]))
        retval[c] = tmp
    return retval


def calc_Normalized_Points(points, normalized_length, shift):
    p = np.array(points)
    p = (p - shift) * normalized_length

    retval = {i: (int(Decimal(v[0]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                  int(Decimal(v[1]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                  int(Decimal(v[2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))) \
                  for i, v in enumerate(p)}
    return retval


def calc_Cell_Geometry(points_in_cell, nrml_points):
    retval = dict()
    for k, points in points_in_cell.items():
        p_max = np.max(np.array([nrml_points[p] for p in points]), axis = 0)
        p_min = np.min(np.array([nrml_points[p] for p in points]), axis = 0)
        center = np.mean([p_max, p_min], axis = 0, dtype = np.int32)
        size = (p_max - p_min)[0]
        retval[k] = (center, size)
    return retval


# ---------------------------------------------------------------------------------------------------------------

def get_Range_of_Subdomains(points):
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


def find_CellCenter_of_FineMultiBlock(c_tuple, v_tuple, cellCenter_tupleSet):
    c = np.array(c_tuple)
    v = np.array(v_tuple)
    size = max(abs(v - c))

    v_fine = {tuple(v + np.array(csys[e]) * size // 4) for e in range(19, 27)}
    v_fine.intersection_update(cellCenter_tupleSet)
    if v_fine:
        return v_fine
    return set()


def find_CellCenter_of_CoarseMultiBlock(c_tuple, v_tuple, cellCenter_tupleSet):
    c = np.array(c_tuple)
    v = np.array(v_tuple)
    size = max(abs(v - c))

    v_coarse = {tuple(v + np.array(csys[e]) * size // 2) for e in range(19, 27)}
    v_coarse.intersection_update(cellCenter_tupleSet)
    if v_coarse:
        return v_coarse
    return set()


def calc_CsysDirection_from_CellCenter(c_tuple, v_tuple):
    c = np.array(c_tuple)
    v = np.array(v_tuple)
    size = max(abs(v - c))
    n = tuple((v - c) // size)
    return csys_v[n]



def const_FluidMesh(pMesh):
    mesh = {tuple(c): lattice() for c in pMesh.cell_center.values()}
    cell_size = {tuple(c): pMesh.cell_size[i] for i, c in pMesh.cell_center.items()}
    for c in mesh:
        mesh[c].size = cell_size[c]


    for boundary in pMesh.boundaries:
        # pprint.pprint(boundary)
        nFaces = boundary["nFaces"]
        startFace = boundary["startFace"]
        boundaryType = boundary["type"]
        boundaryName = boundary["name"]

        for f in range(startFace, startFace + nFaces):
            ownerCell = pMesh.get_ownerCellNo_of_Face(f)
            isys = pMesh.calc_FaceSys_of_CellNo(ownerCell, f)
            c = pMesh.trans_CellNo_to_Cell_Center(ownerCell)
            mesh[c].boundaryType[isys] = boundaryType
            mesh[c].boundaryName[isys] = boundaryName


    for c in mesh:
        boundaryType = mesh[c].boundaryType
        boundaryName = mesh[c].boundaryName

        if boundaryType:
            keys = [k for k in boundaryType if boundaryType[k] != "processor"]
            mesh[c].boundaryType = {i: boundaryType[k] for k in keys for i in ref_faces[k]}
            mesh[c].boundaryName = {i: boundaryName[k] for k in keys for i in ref_faces[k]}

        if "wall" in boundaryType.values():
            keys = [k for k in boundaryType if boundaryType[k] == "wall"]
            mesh[c].boundaryType = {i: "wall" for k in keys for i in ref_faces[k]}
            mesh[c].boundaryName = {i: boundaryName[k] for k in keys for i in ref_faces[k]}


    cell_center_Set = {tuple(c) for c in pMesh.cell_center.values()}
    max_cellSize = max({c.size for c in mesh.values()})
    min_cellSize = min({c.size for c in mesh.values()})


# MultiBlockの境界の場合ひとまず"unknown"が割り振られる
    for c in mesh:
        size = mesh[c].size
        keys = [i for i in range(1, 27) if i not in mesh[c].boundaryType]

        for k in keys:
            v = tuple(np.array(c) + np.array(csys[k]) * size)
            if v not in cell_center_Set:
                mesh[c].boundaryType[k] = "unknown"
                mesh[c].boundaryName[k] = "unknown"

                # if max_cellSize == min_cellSize:
                #     continue

                # if size == max_cellSize:
                #     vf = find_CellCenter_of_FineMultiBlock(c, v, cell_center_Set)
                #     if vf:
                #         mesh[c].boundaryType[k] = "mb_fine" #相手がfine
                #         mesh[c].boundaryName[k] = "mb_fine"
                #         continue

                # if size == min_cellSize:
                #     vc = find_CellCenter_of_CoarseMultiBlock(c, v, cell_center_Set)
                #     if vc:
                #         n = calc_CsysDirection_from_CellCenter(c, v)
                #         n = csys_inv[n]
                #         mesh[c].boundaryType[k] = "mb_coarse" #相手がcoarse
                #         mesh[c].boundaryName[k] = "mb_coarse"
                #         continue

                # vf = find_CellCenter_of_FineMultiBlock(c, v, cell_center_Set)
                # if vf:
                #     mesh[c].boundaryType[k] = "mb_fine"
                #     mesh[c].boundaryName[k] = "mb_fine"
                #     continue

                # vc = find_CellCenter_of_CoarseMultiBlock(c, v, cell_center_Set)
                # if vc:
                #     n = calc_CsysDirection_from_CellCenter(c, v)
                #     n = csys_inv[n]
                #     mesh[c].boundaryType[k] = "mb_coarse"
                #     mesh[c].boundaryName[k] = "mb_coarse"
                #     continue

    return mesh




def update_boundaryType(processor, mesh, other_mesh, domainRange_dict):
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : tuple[int] of size of the levels

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids
    np.ndarray[np.int64]
        obs_group_ids
    """

    domainRange = domainRange_dict[processor]
    other_Range = {proc: domainRange_dict[proc] for proc in domainRange_dict if not proc == processor}
 # 自身のprocessorの領域のBounding Boxの範囲
    p_xmin = domainRange[0][0] - 1
    p_xmax = domainRange[1][0] + 1
    p_ymin = domainRange[0][1] - 1
    p_ymax = domainRange[1][1] + 1
    p_zmin = domainRange[0][2] - 1
    p_zmax = domainRange[1][2] + 1

# 接触している可能性がある領域番号をリストに格納
    contact_procList = []
    for proc in other_Range:
        q_xmin = other_Range[proc][0][0]
        q_xmax = other_Range[proc][1][0]
        q_ymin = other_Range[proc][0][1]
        q_ymax = other_Range[proc][1][1]
        q_zmin = other_Range[proc][0][2]
        q_zmax = other_Range[proc][1][2]

        if (p_xmax > q_xmin) and (p_xmin < q_xmax) and \
           (p_ymax > q_ymin) and (p_ymin < q_ymax) and \
           (p_zmax > q_zmin) and (p_zmin < q_zmax):
            contact_procList.append(proc)

# 接触している可能性がある領域に含まれる各セルの最大頂点座標と最小頂点座標を辞書に格納
    other_pmin = {proc: np.array([np.array(c) + np.array(csys[25]) * (other_mesh[proc][c].size // 2)
                                  for c in other_mesh[proc]]) for proc in contact_procList}
    other_pmax = {proc: np.array([np.array(c) + np.array(csys[19]) * (other_mesh[proc][c].size // 2)
                                  for c in other_mesh[proc]]) for proc in contact_procList}



    tmpDict = {c: {k: v for k, v in mesh[c].boundaryType.items()
                if v == "processor" or v == "unknown"}
                for c in mesh
                if "processor" in mesh[c].boundaryType.values() or
                     "unknown" in mesh[c].boundaryType.values()}


    cell_center_Set = {tuple(c) for c in mesh}
    max_cellSize = max({c.size for c in mesh.values()})
    min_cellSize = min({c.size for c in mesh.values()})


    for c, d in tmpDict.items():
        size = mesh[c].size

        tmpList = [k for k, boundaryType in d.items() if boundaryType == "unknown"]
        for k in tmpList:
            v = tuple(np.array(c) + np.array(csys[k]) * size)

            if size == max_cellSize:
                vf = find_CellCenter_of_FineMultiBlock(c, v, cell_center_Set)
                if vf:
                    mesh[c].boundaryType[k] = "mb_fine" #相手がfine
                    mesh[c].boundaryName[k] = tuple(vf)
                    continue

            if size == min_cellSize:
                vc = find_CellCenter_of_CoarseMultiBlock(c, v, cell_center_Set)
                if vc:
                    p = tuple(vc)[0]
                    n = calc_CsysDirection_from_CellCenter(v, p)
                    mesh[c].boundaryType[k] = "mb_coarse" #相手がcoarse
                    mesh[c].boundaryName[k] = (p, n)
                    continue

            vf = find_CellCenter_of_FineMultiBlock(c, v, cell_center_Set)
            if vf:
                mesh[c].boundaryType[k] = "mb_fine"
                mesh[c].boundaryName[k] = tuple(vf)
                continue

            vc = find_CellCenter_of_CoarseMultiBlock(c, v, cell_center_Set)
            if vc:
                p = tuple(vc)[0]
                n = calc_CsysDirection_from_CellCenter(v, p)
                mesh[c].boundaryType[k] = "mb_coarse"
                mesh[c].boundaryName[k] = (p, n)
                continue

            for proc in contact_procList:
                if v in other_mesh[proc]:
                    mesh[c].boundaryType[k] = "processor"
                    mesh[c].boundaryName[k] = (v, proc)
                    break

                n = np.array(v)
                pmin = other_pmin[proc]
                pmax = other_pmax[proc]
                isInclude = np.stack([np.all(n >= pmin, axis = 1), np.all(n <= pmax, axis = 1)], axis = 1)

                if np.any(np.all(isInclude, axis = 1)):
                    cell_center_Set = {c for c in other_mesh[proc]}

                    vf = find_CellCenter_of_FineMultiBlock(c, v, cell_center_Set)
                    if vf:
                        mesh[c].boundaryType[k] = "processor_mb_fine" #相手がfine
                        mesh[c].boundaryName[k] = (tuple(vf), proc)
                        break

                    vc = find_CellCenter_of_CoarseMultiBlock(c, v, cell_center_Set)
                    if vc:
                        p = tuple(vc)[0]
                        n = calc_CsysDirection_from_CellCenter(v, p)
                        mesh[c].boundaryType[k] = "processor_mb_coarse" #相手がcoarse
                        mesh[c].boundaryName[k] = (p, n, proc)
                        break

            else: # When "nb" was nowhere to be found.
                mesh[c].boundaryType[k] = "wall"
                mesh[c].boundaryName[k] = "wall_temporary"


        tmpList = [k for k, boundaryType in d.items() if boundaryType == "processor"]
        for k in tmpList:
            proc = mesh[c].boundaryName[k]
            proc = re.sub(r'^procBoundary\d+to', 'processor', proc)
            v = tuple(np.array(c) + np.array(csys[k]) * size)

            if v in other_mesh[proc]:
                mesh[c].boundaryType[k] = "processor"
                mesh[c].boundaryName[k] = (v, proc)
                continue

            cell_center_Set = {c for c in other_mesh[proc]}

            vf = find_CellCenter_of_FineMultiBlock(c, v, cell_center_Set)
            if vf:
                mesh[c].boundaryType[k] = "processor_mb_fine" #相手がfine
                mesh[c].boundaryName[k] = (tuple(vf), proc)
                continue

            vc = find_CellCenter_of_CoarseMultiBlock(c, v, cell_center_Set)
            if vc:
                p = tuple(vc)[0]
                n = calc_CsysDirection_from_CellCenter(v, p)
                mesh[c].boundaryType[k] = "processor_mb_coarse" #相手がcoarse
                mesh[c].boundaryName[k] = (p, n, proc)
                continue

    return mesh



def tables_CellCenter_to_CellNo(processor, lat):

    tmpList = [{"p":c , "size": lat[c].size} for c in lat]
    tmpList.sort(key = itemgetter("size"), reverse = True)
    cellNo = {_["p"]: i for i, _ in enumerate(tmpList)}

    ghost = dict()
    for c in lat:
        size = lat[c].size
        boundary_direction = lat[c].boundaryType.keys()

        for k in boundary_direction:
            g = tuple(np.array(c) + np.array(csys[k]) * size)
            if g not in ghost:
                ghost[g] = lattice()
                ghost[g].size = size
                n = csys_inv[k]
                ghost[g].boundaryType[n] = lat[c].boundaryType[k]
                ghost[g].boundaryName[n] = lat[c].boundaryName[k]
            else:
                n = csys_inv[k]
                ghost[g].boundaryType[n] = lat[c].boundaryType[k]
                ghost[g].boundaryName[n] = lat[c].boundaryName[k]



    latLength = len(lat)
    tmpList =[{"p":c , "size": ghost[c].size} for c in ghost]
    tmpList.sort(key = itemgetter("size"), reverse = True)
    gcellNo = {_["p"]: i + latLength for i, _ in enumerate(tmpList)}

    ret_dict = {"fluid": cellNo, "bound": gcellNo}

    return ret_dict



def generate_GhostCell(processor, lat, table, other_table):

    ghost = dict()
    for c in lat:
        boundary_direction = lat[c].boundaryType.keys()
        size = lat[c].size

        for k in boundary_direction:
            g = tuple(np.array(c) + np.array(csys[k]) * size)

            if g not in ghost:
                ghost[g] = lattice()
                ghost[g].size = size
                n = csys_inv[k]
                ghost[g].boundaryType[n] = lat[c].boundaryType[k]
                ghost[g].boundaryName[n] = lat[c].boundaryName[k]

            else:
                n = csys_inv[k]
                ghost[g].boundaryType[n] = lat[c].boundaryType[k]
                ghost[g].boundaryName[n] = lat[c].boundaryName[k]


    # print(ghost[(180,4,4)].boundaryType)
    # print(ghost[(180,4,4)].boundaryName)



    latLength = len(lat)

    cellNo = table["fluid"]
    gcellNo = table["bound"]


    tmpSet = {i for i in range(1, 27)}
    cellCenter = list(lat.keys())

    for c in cellCenter:
        n = cellNo[c]
        lat[n] = lat.pop(c)

        boundSet = set(lat[n].boundaryType.keys())
        fluidSet = tmpSet.difference(boundSet)
        neighbourCell = {k:  cellNo[lat[n].get_neighbourPoint(c, k)] for k in fluidSet} | \
                        {k: gcellNo[lat[n].get_neighbourPoint(c, k)] for k in boundSet}
        lat[n].neighbourCell = neighbourCell
        lat[n].center = c


    import pandas as pd

    tmp_dict = dict()
    for n, v in lat.items():
        data = {
            "id": n,
             "x": v.center[0],
             "y": v.center[1],
             "z": v.center[2],
             "size": v.size
        } | {
            k: v.neighbourCell[k] for k in range(1,27)
        }
        tmp_dict[n] = pd.Series(data = data)

    df = pd.DataFrame.from_dict(tmp_dict, orient = "index")
    df = df.set_index("id")



    boundaryTypeList = {b for c in ghost for _, b in ghost[c].boundaryType.items()}
    boundaryTypeList = sorted(list(boundaryTypeList))


    boundary_df = dict()
    for boundaryType in boundaryTypeList:
        if boundaryType == "processor":
            tmpList = [
                {
                    "id": gcellNo[c],
                    "size": ghost[c].size,
                    "proc": v[1],
                    "target": other_table[v[1]]["fluid"][c],
                    "x": c[0],
                    "y": c[1],
                    "z": c[2]
                }
                for c in ghost for k, v in ghost[c].boundaryName.items() if ghost[c].boundaryType[k] == boundaryType
            ]
            tmpList.sort(key = itemgetter("id", "size", "size"), reverse = False)
            tmp_dict = {n: pd.Series(data = v) for n, v in enumerate(tmpList)}
            boundary_df[boundaryType] = pd.DataFrame.from_dict(tmp_dict, orient = "index")
            boundary_df[boundaryType] = boundary_df[boundaryType].set_index("id").drop_duplicates()
            continue

        proc = processor
        n = 0
        tmpList = list()

        for c in ghost:
            for k, b in ghost[c].boundaryType.items():
                if b != boundaryType: continue
                tmpList.append(
                    {"id": gcellNo[c],
                     "size": ghost[c].size,
                     "k": k,
                     "proc": proc,
                     "target": n,
                     "x": c[0],
                     "y": c[1],
                     "z": c[2]
                })

        tmpList.sort(key = itemgetter("id", "size", "k"), reverse = False)

        tmp_dict = {n: pd.Series(data = v) for n, v in enumerate(tmpList)}
        boundary_df[boundaryType] = pd.DataFrame.from_dict(tmp_dict, orient = "index")
        boundary_df[boundaryType] = boundary_df[boundaryType].set_index("id")


    df.to_csv(processor + "/neighbour.csv")
    # print("")
    # print(processor)
    for boundaryType in boundaryTypeList:
        # print(boundaryType)
        boundary_df[boundaryType].to_csv(processor + "/" + boundaryType + ".csv")
        # pprint.pprint(boundary_df[boundaryType])


    # if processor == "processor1":
    #     pprint.pprint(df)
    #     for boundaryType in boundaryTypeList:
    #         print("")
    #         print(boundaryType)
    #         pprint.pprint(boundary_df[boundaryType])



    # print(gcellNo[(180, 4, -4)])
    # print(ghost[(180, 4, -4)].size)
    # print(ghost[(180, 4, -4)].boundaryType)
    # print(ghost[(180, 4, -4)].boundaryName)







    mesh = dict()
    # center_dict = {lat[c].id: c for c in lat}
    # size_dict = {lat[c].id: lat[c].size for c in lat}
    # btype_dict = {lat[c].id: lat[c].boundaryType for c in lat}
    # bname_dict = {lat[c].id: lat[c].boundaryName for c in lat}
    # tmpSet = {i for i in range(1, 27)}

    # for i in range(latLength):
    #     mesh[i] = lattice()
    #     mesh[i].center = center_dict[i]
    #     mesh[i].size = size_dict[i]
    #     mesh[i].boundaryType = btype_dict[i]
    #     mesh[i].boundaryName = bname_dict[i]

    #     c = mesh[i].center
    #     boundSet = set(mesh[i].boundaryType.keys())
    #     fluidSet = tmpSet.difference(boundSet)
    #     neighbourCell = {k: lat[mesh[i].neibp(c, k)].id for k in fluidSet} | \
    #                     {k: ghost[mesh[i].neibp(c, k)].id for k in boundSet}
    #     mesh[i].neighbourCell = neighbourCell


    # center_dict = {ghost[c].id: c for c in ghost}
    # size_dict = {ghost[c].id: ghost[c].size for c in ghost}
    # btype_dict = {ghost[c].id: ghost[c].boundaryType for c in ghost}
    # bname_dict = {ghost[c].id: ghost[c].boundaryName for c in ghost}

    # for i in range(latLength, ghostLength):
    #     mesh[i] = lattice()
    #     mesh[i].center = center_dict[i]
    #     mesh[i].size = size_dict[i]
    #     mesh[i].boundaryType = btype_dict[i]
    #     mesh[i].boundaryName = bname_dict[i]

    #     c = mesh[i].center
    #     boundSet = set(mesh[i].boundaryType.keys())
    #     neighbourCell = {k: lat[mesh[i].neibp(c, k)].id for k in boundSet}
    #     mesh[i].neighbourCell = neighbourCell

    return mesh
