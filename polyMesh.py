
from read_polyMesh import \
    read_PointsFile, \
    read_FacesFile, \
    read_CellFaceFile, \
    read_CellLevelFile, \
    read_BoundaryFile, \
    read_Range_of_Subdomains
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from operator import itemgetter
import re
import pprint
import pandas as pd

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

# ---------------------------------------------------------------------------------------------------------------


def SubdomainRange(processor, range_shared):
    points = read_PointsFile(processor)
    cellLevel = read_CellLevelFile(processor)
    min_SubdomainRange, max_SubdomainRange = read_Range_of_Subdomains(points)
    xmin, ymin, zmin = min_SubdomainRange
    xmax, ymax, zmax = max_SubdomainRange
    range_shared[processor] = {"min": (xmin, ymin, zmin), "max": (xmax, ymax, zmax), "max_cellLevel": max(cellLevel)}

    # np.set_printoptions(precision = 8, floatmode = "fixed", suppress = True)
    # print(f"\t{processor}: min: {np.array(min_SubdomainRange)}, max: {np.array(max_SubdomainRange)}")


def import_polyMesh(processor, base_size, range_shared, master_shared):
    points = read_PointsFile(processor)
    faces = read_FacesFile(processor)
    owner = read_CellFaceFile(processor, "owner")
    neighbour = read_CellFaceFile(processor, "neighbour")
    boundaries = read_BoundaryFile(processor)

    max_cellLevel = np.max(np.array([v["max_cellLevel"] for v in range_shared.values()]))
    normalized_Length = (2 ** (max_cellLevel + 1)) / base_size
    shift = range_shared["processor0"]["min"]

    p = np.array(points)
    p = (p - shift) * normalized_Length
    normalized_points = {i: np.array(
                         (int(Decimal(v[0]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                          int(Decimal(v[1]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                          int(Decimal(v[2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))), \
                          dtype = np.int32) for i, v in enumerate(p)}

    range_proc = range_shared[processor]
    nmin_p = (np.array(range_proc["min"]) - shift) * normalized_Length
    nmax_p = (np.array(range_proc["max"]) - shift) * normalized_Length
    nmin_x = int(Decimal(nmin_p[0]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    nmin_y = int(Decimal(nmin_p[1]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    nmin_z = int(Decimal(nmin_p[2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    nmax_x = int(Decimal(nmax_p[0]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    nmax_y = int(Decimal(nmax_p[1]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    nmax_z = int(Decimal(nmax_p[2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    nmin = (nmin_x, nmin_y, nmin_z)
    nmax = (nmax_x, nmax_y, nmax_z)
 
    range_dict = range_shared[processor]
    range_dict["nmin"] = nmin
    range_dict["nmax"] = nmax
    range_dict["normalized_Length"] = normalized_Length
    range_dict["shift"] = shift
    range_shared[processor] = range_dict
    np.set_printoptions(precision = 8, floatmode = "fixed", suppress = True)
    print(f"\t{processor}: min: {range_dict["nmin"]}, max: {range_dict["nmax"]}")


    owner_face = np.arange(len(owner), dtype = np.int32)
    neighbour_face = np.arange(len(neighbour), dtype = np.int32)
    face_list = np.append(owner_face, neighbour_face)

    owner_cell = np.array(owner, dtype = np.int32)
    neighbour_cell = np.array(neighbour, dtype = np.int32)
    cell_list = np.append(owner_cell, neighbour_cell)

    Master_df = pd.DataFrame(
        {
            "cell": cell_list,
            "face": face_list,
        }
    )

    unique_face_list = np.unique(face_list)
    unique_face_point_dict = {n: np.array([normalized_points[p] for p in faces[n]]) for n in unique_face_list}
    unique_face_minmax_dict = {n: np.append(np.min(p, axis = 0), np.max(p, axis = 0)) for n, p in unique_face_point_dict.items()}
    face_minmax_list = np.array([unique_face_minmax_dict[n] for n in face_list])

    Master_df["fxmin"] = face_minmax_list[:, 0]
    Master_df["fymin"] = face_minmax_list[:, 1]
    Master_df["fzmin"] = face_minmax_list[:, 2]
    Master_df["fxmax"] = face_minmax_list[:, 3]
    Master_df["fymax"] = face_minmax_list[:, 4]
    Master_df["fzmax"] = face_minmax_list[:, 5]

    pmin_df = Master_df[["cell", "fxmin", "fymin", "fzmin"]].groupby("cell").min()
    pmax_df = Master_df[["cell", "fxmax", "fymax", "fzmax"]].groupby("cell").max()

    pmin = np.array(pmin_df)
    pmax = np.array(pmax_df)
    pmean = np.array([pmin, pmax]).mean(axis = 0, dtype = np.int32)
    size = (pmax - pmin)[:, 0]

    Master_df["size"] = np.array([size[n] for n in cell_list])
    Master_df["nx"] = np.array([pmean[n] for n in cell_list])[:, 0]
    Master_df["ny"] = np.array([pmean[n] for n in cell_list])[:, 1]
    Master_df["nz"] = np.array([pmean[n] for n in cell_list])[:, 2]

    def fcsys(pn, fmin, fmax):
        bl = fmin == fmax
        nc = pn[bl][0]
        nf = fmin[bl][0]
        sys = np.array([[1, 3], [2, 4], [5, 6]])
        return sys[bl][0][[nc < nf, nf < nc]][0]

    pn = np.array(Master_df[["nx", "ny", "nz"]])
    fmin = np.array(Master_df[["fxmin", "fymin", "fzmin"]])
    fmax = np.array(Master_df[["fxmax", "fymax", "fzmax"]])

    fcsys_list = np.array([fcsys(p, fm, fx) for p, fm, fx in zip(pn, fmin, fmax)], dtype = np.int32)
    Master_df["fcsys"] = fcsys_list


    nInternalFaces = len(neighbour)
    nFaces = np.array([v["nFaces"] for v in boundaries], dtype = np.int32)
    startFace = np.array([v["startFace"] for v in boundaries], dtype = np.int32)
    endFace = startFace + nFaces
    boundaryType = [v["type"] for v in boundaries]
    boundaryName = [v["name"] for v in boundaries]

    def face_type(faceNo, pn, fmin, fmax):
        post_fix = ""
        if np.any(fmin == pn) or np.any(fmax == pn): post_fix = "_mbFine"
        if faceNo < nInternalFaces:
            return ["InternalFace" + post_fix, "None"]
        k = np.argmax(np.all([startFace <= faceNo, faceNo < endFace], axis = 0))
        return [boundaryType[k] + post_fix, boundaryName[k]]

    face_type_list = np.array(list(map(face_type, face_list, pn, fmin, fmax)))
    Master_df["boundaryType"] = pd.Series(face_type_list[:, 0], dtype = "string")
    Master_df["boundaryName"] = pd.Series(face_type_list[:, 1], dtype = "string")


    is_mbFine = np.array(Master_df["boundaryType"] == "InternalFace_mbFine")
    face_mbFine = np.array(Master_df["face"][is_mbFine])
    _ = face_mbFine.reshape([len(face_mbFine), -1])
    is_duplicate_face = np.any(face_list == _, axis = 0)
    is_mbCoarse = is_duplicate_face ^ is_mbFine
    before = Master_df["boundaryType"].tolist()
    after = np.where(is_mbCoarse, "InternalFace_mbCoarse", before)
    Master_df["boundaryType"] = pd.Series(after, dtype = "string")

    Master_df["processor"] = int(processor.replace("processor", ""))
    Master_df["processor"] = Master_df["processor"].astype("int32")

    master_shared[processor] = Master_df


def find_processor_mbFine(processor, master_shared):

    Master_df = master_shared[processor]
    other_master_dict = {proc: df for proc, df in master_shared.items() if not proc == processor}

    findex = ["fxmin", "fymin", "fzmin", "fxmax", "fymax", "fzmax", "size"]
    boundaryType = np.array(Master_df["boundaryType"])
    boundaryName = np.array(Master_df["boundaryName"])
    is_processor = boundaryType == "processor"
    p = re.compile("[a-zA-Z]+[0-9]+to")
    processorName = [p.sub("processor", n) for n in boundaryName[is_processor]]
    processor_findex = np.array(Master_df[findex])[is_processor]
    processor_findex[:,6] = 2 * processor_findex[:,6]
    other_processor_findex = {n: np.array(df[df["boundaryType"] == "processor_mbFine"][findex]) for n, df in other_master_dict.items()}
    is_mbCoarse = [findex in other_processor_findex[proc] for proc, findex in zip(processorName, processor_findex)]

    boundaryType[is_processor] = np.where(is_mbCoarse, "processor_mbCoarse", "processor")
    Master_df["boundaryType"] = boundaryType
    master_shared[processor] = Master_df


def generate_LinkWiseLattice(processor, master_shared, meshes_shared, ghosts_shared):

    Master_df = master_shared[processor]

    size_Series = np.array(Master_df["size"])
    pn_Series = np.array(Master_df[["nx", "ny", "nz"]])
    cell_Series = np.array(Master_df["cell"])

    unique_cell_index, indices = np.unique(cell_Series, return_index = True)
    size = size_Series[indices]
    pn = pn_Series[indices]

    amp = size.reshape([len(size), 1])
    pcsys_dict = {n: pn + (amp * np.array(v)) for n, v in csys.items()}
    tmp_list = np.concatenate(list(pcsys_dict.values()))
    tmp_list = np.concatenate([pn, tmp_list])
    tmp_list = np.array(pd.DataFrame(data = tmp_list).drop_duplicates())
    tmp_list = np.concatenate([pn, tmp_list])
    pn_ghostCell_df = pd.DataFrame(data = tmp_list).drop_duplicates(keep = False)
    ghostCell = np.array(pn_ghostCell_df)

    ns = np.max(unique_cell_index) + 1
    ne = ns + len(pn_ghostCell_df)
    unique_ghost_index = np.arange(ns, ne, dtype = np.int32)

    tmp_cell = np.concatenate([pn, ghostCell])
    tmp_index = np.append(unique_cell_index, unique_ghost_index)
    pinv = {tuple(v): n for n, v in zip(tmp_index, tmp_cell)}
    pcsys_tuple = tuple(map(tuple, np.concatenate([pcsys_dict[k] for k in np.arange(1,27)])))

    ncell = len(unique_cell_index)
    csys_cell_index = np.array([pinv[v] for v in pcsys_tuple], dtype = np.int32).reshape([ncell, -1], order = "F")
    csys_cell_df = pd.DataFrame(csys_cell_index)
    csys_cell_df = csys_cell_df.set_axis(["csys_" + str(i) for i in range(1,27)], axis = "columns")
    csys_cell_df["cell"] = unique_cell_index

    Link_Wise_Lattice = pd.DataFrame(
        {
            "cell": unique_cell_index,
            "size": size,
            "nx": pn[:,0],
            "ny": pn[:,1],
            "nz": pn[:,2],
        }
    )

    Link_Wise_Lattice = Link_Wise_Lattice.merge(csys_cell_df)
    meshes_shared[processor] = Link_Wise_Lattice

    pn_ghostCell_df = pn_ghostCell_df.set_axis(["nx", "ny", "nx"], axis = "columns")
    pn_ghostCell_df["ghostCell"] = unique_ghost_index
    ghosts_shared[processor] = pn_ghostCell_df


def boundary_condition(processor, master_shared, meshes_shared, ghosts_shared, range_shared):
    Lattice_df = meshes_shared[processor]
    Master_df = master_shared[processor]
    pn_ghost_df = ghosts_shared[processor]

    csys_array = np.array(Lattice_df.filter(like = "csys_", axis = 1))
    # size = np.array(Lattice_df["size"])
    # pn = np.array(Lattice_df[["nx", "ny", "nz"]])
    # unique_innerCell_list = np.array(Lattice_df["cell"])
    # n_innerCell = np.max(unique_innerCell_list)
    unique_ghostCell_list = np.array(pn_ghost_df["ghostCell"])


    csys_array_flatten = csys_array.flatten()
    _dict = {k: list() for k in csys_array_flatten}
    [_dict[k].append(i) for i, k in enumerate(csys_array_flatten)]
    _list = np.array([np.divmod(ks, 26, dtype = np.int32) for gn in unique_ghostCell_list for ks in _dict[gn]])
    _cell = _list[:, 0]
    _csys = np.array([csys_inv[k + 1] for k in _list[:, 1]])
    _ghost = np.array([gn for gn in unique_ghostCell_list for _ in _dict[gn]])
    _unknown_df = pd.DataFrame(
        {
            "ghostCell": _ghost,
            "cell": _cell,
            "csys": _csys,
        }
    )
    _unknown_df["boundaryType"] = "unknown"
    _unknown_df["boundaryName"] = "unknown"


    _ = Master_df[["cell", "fcsys", "boundaryType", "boundaryName"]].query("boundaryType != 'InternalFace'")
    cells = np.array(_["cell"])
    fcsys = np.array(_["fcsys"])
    boundaryTypes = np.array(_["boundaryType"])
    boundaryNames = np.array(_["boundaryName"])
    _list = [[csys_array[n][k-1], n, csys_inv[k], boundaryType, boundaryName]
             for n, k, boundaryType, boundaryName in zip(cells, fcsys, boundaryTypes, boundaryNames)] 
    _df = pd.DataFrame(_list).set_axis(["ghostCell", "cell", "csys", "boundaryType", "boundaryName"], axis = "columns")


    ref_boundaryType = ["processor", "InternalFace_mbFine", "InternalFace_mbCoarse", "processor_mbFine", "processor_mbCoarse"]
    ref_isin = _df["boundaryType"].isin(ref_boundaryType)
    _ref_df = _df[ref_isin].drop_duplicates(subset="ghostCell")
    # _ref_df = _df.query(
    #       "boundaryType == 'processor' \
    #     or boundaryType == 'InternalFace_mbFine' \
    #     or boundaryType == 'InternalFace_mbCoarse' \
    #     or boundaryType == 'processor_mbFine' \
    #     or boundaryType == 'processor_mbCoarse'"
    # ).drop_duplicates(subset="ghostCell")
    _unknown_list = np.array(_unknown_df["ghostCell"])
    _list = np.array(_ref_df["ghostCell"])
    drop_indexes = np.isin(_unknown_list, _list, invert=True)
    _unknown_df = _unknown_df[drop_indexes]


    bnd_isin = ~ref_isin
    _bnd_df = _df[bnd_isin].drop_duplicates(subset="ghostCell")
    # _bnd_df = _df.query(
    #       "boundaryType != 'processor' \
    #     or boundaryType != 'InternalFace_mbFine' \
    #     or boundaryType != 'InternalFace_mbCoarse' \
    #     or boundaryType != 'processor_mbFine' \
    #     or boundaryType != 'processor_mbCoarse'"
    # ).drop_duplicates(subset="ghostCell")
    _unknown_list = np.array(_unknown_df["ghostCell"])
    _list = np.array(_bnd_df["ghostCell"])
    bnd_indexes = np.isin(_unknown_list, _list)
    _unknown_bnd_df = _unknown_df[bnd_indexes]
    ref_indexes = ~bnd_indexes
    _unknown_ref_df = _unknown_df[ref_indexes]


    unique_unknown_ref_list = np.unique(np.array(_unknown_ref_df["ghostCell"]))
    _pn = {gn: (nx, ny, nz) for nx, ny, nz, gn in np.array(pn_ghost_df)}
    _pn_unknown = {_pn[gn]: gn for gn in unique_unknown_ref_list}


    all_processor = list(range_shared.keys())
    proc_range = {proc: d for proc, d in range_shared.items()}
    this_range = proc_range[processor]
    nmin = this_range["nmin"]
    nmax = this_range["nmax"]
    pnmin = {proc: d["nmin"] for proc, d in proc_range.items()}
    pnmax = {proc: d["nmax"] for proc, d in proc_range.items()}

    contact_procList = [proc for proc in all_processor
                        if (nmax[0] >= pnmin[proc][0]) and (nmin[0] <= pnmax[proc][0]) and \
                           (nmax[1] >= pnmin[proc][1]) and (nmin[1] <= pnmax[proc][1]) and \
                           (nmax[2] >= pnmin[proc][2]) and (nmin[2] <= pnmax[proc][2])
                        ]

    print(contact_procList)
    all_master_df = [master_shared[proc] for proc in contact_procList]
    # all_master_df = [df for df in master_shared.values()]
    all_pmin = np.concatenate([np.array(df[["cell", "fxmin", "fymin", "fzmin"]].groupby("cell").min()) for df in all_master_df])
    all_pmax = np.concatenate([np.array(df[["cell", "fxmax", "fymax", "fzmax"]].groupby("cell").max()) for df in all_master_df])
    all_proc = np.concatenate([np.array(df[["cell", "processor"]].groupby("cell", as_index = False).max()) for df in all_master_df])

    # is_included = np.array(
    #     [np.all([all_pmin[:, 0] <= gn[0], gn[0] <= all_pmax[:, 0],
    #              all_pmin[:, 1] <= gn[1], gn[1] <= all_pmax[:, 1],
    #              all_pmin[:, 2] <= gn[2], gn[2] <= all_pmax[:, 2]
    #              ], axis = 0) for gn in _pn_unknown])
    # all_cell_index = all_proc[:, 0]
    # all_proc_index = all_proc[:, 1]
    # cell_related_unDefined_GCell = [all_cell_index[isin] for isin in is_included]
    # proc_related_unDefined_GCell = [all_proc_index[isin] for isin in is_included]





    if processor == "processor0":
        print("")
        # print(proc_pmin)
        # print(proc_pmax)
        # print(_internal_mbFine_df)
        # print(_internal_mbCoarse_df)
        # print(_processor_mbFine_df)
        # print(_processor_mbCoarse_df)
        # print(_unknown_df)



    # ghost_df = pd.concat([_df, ghost_df], axis=0).drop_duplicates(subset=["ghostCell", "cell", "csys"], keep="first")
    # _list = np.unique(np.array(face_contact_ghostCell_df["ghostCell"]))
    # _list = np.append(_list, unique_ghostCell_list)
    # edge_contact_ghostCell = np.array(pd.Series(data = _list).drop_duplicates(keep = False))


 









    # _list = csys_array[:, :6]
    # face_contact_ghostCell = np.unique(_list[_list > n_innerCell])
    # _list = [np.argwhere(csys_array == gc)[0] for gc in face_contact_ghostCell]
    # pn_face_contact_ghostCell = np.array([pn[n] + (size[n] * np.array(csys[i + 1])) for n, i in _list])

    # _list = np.append(face_contact_ghostCell, unique_ghostCell_list)
    # edge_contact_ghostCell = np.array(pd.Series(data = _list).drop_duplicates(keep = False))
    # _list = [np.argwhere(csys_array == gc)[0] for gc in edge_contact_ghostCell]
    # pn_edge_contact_ghostCell = np.array([pn[n] + (size[n] * np.array(csys[i + 1])) for n, i in _list])

    # _list = csys_array[:, 6:]
    # _list = np.unique(_list[_list > n_innerCell])
    # intersect_contact_ghostCell = np.intersect1d(_list, face_contact_ghostCell)
    # _list = [np.argwhere(csys_array == gc)[0] for gc in intersect_contact_ghostCell]
    # pn_intersect_contact_ghostCell = np.array([pn[n] + (size[n] * np.array(csys[i + 1])) for n, i in _list])




    # all_cell_index = all_proc[:, 0]
    # all_proc_index = all_proc[:, 1]
    # cell_related_unDefined_GCell = [all_cell_index[isin] for isin in is_included]
    # proc_related_unDefined_GCell = [all_proc_index[isin] for isin in is_included]

    # if processor == "processor0":
    #     print("")
    #     print(ghost_df)
        # print(ghost_pn_df)
        # print(len(unique_ghostCell_list), len(face_contact_ghostCell), len(edge_contact_ghostCell), len(intersect_contact_ghostCell))
        # findex =np.array(face_contact_ghostCell_df["ghostCell"])
        # for n in findex:
        #     if not n in unique_ghostCell_list: print(n)
        # print(len(np.unique(np.array(face_contact_ghostCell_df["ghostCell"]))))
        # print(len(edge_contanct_ghostCell))
        # print(tmp_df)
        # pprint.pprint(cell_related_unDefined_GCell)







if __name__ == '__main__':
    from multiprocessing import Process, Manager, Array, Value, Lock
    import logging
    import time

    logger = logging.getLogger(__name__)
    # logger.propagate = False
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(format)
    # handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    # logging.basicConfig()

    # logger.debug('Debug')
    # logger.info('Informarion')
    # logger.warning('Warning')
    # logger.error('Error')
    # logger.critical('Critical')


    base_size = 1.0
    nProc = 4
    print()


    # print(f"Importing OpenFOAM polyMesh ...", end=" ")

    processor_list = ["processor" + str(i) for i in range(nProc)]
    manager = Manager()

    print(f"Importing OpenFOAM polyMesh ...")

    range_shared = manager.dict()
    process_list = list()
    for processor in processor_list:
        process = Process(
            target = SubdomainRange,
            kwargs = {
                "processor": processor,
                "range_shared": range_shared,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


    start = time.perf_counter()
    # shift = range_shared["processor0"]["min"]
    # max_cellLevel = range_shared["processor0"]["max_cellLevel"]
    master_shared = manager.dict()
    process_list = list()
    for processor in processor_list:
        process = Process(
            target = import_polyMesh,
            kwargs = {
                "processor": processor,
                "base_size": base_size,
                "range_shared": range_shared,
                "master_shared": master_shared,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


    process_list = list()
    # for i in range(nProc):
    for processor in processor_list:
        # processor = "processor" + str(i)
        process = Process(
            target = find_processor_mbFine,
            kwargs = {
                "processor": processor,
                "master_shared": master_shared,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


    meshes_shared = manager.dict()
    ghosts_shared = manager.dict()
    process_list = list()
    # for i in range(nProc):
    for processor in processor_list:
        # processor = "processor" + str(i)
        process = Process(
            target = generate_LinkWiseLattice,
            kwargs = {
                "processor": processor,
                "master_shared": master_shared,
                "meshes_shared": meshes_shared,
                "ghosts_shared": ghosts_shared,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


    process_list = list()
    # for i in range(nProc):
    for processor in processor_list:
        # processor = "processor" + str(i)
        process = Process(
            target = boundary_condition,
            kwargs = {
                "processor": processor,
                "master_shared": master_shared,
                "meshes_shared": meshes_shared,
                "ghosts_shared": ghosts_shared,
                "range_shared": range_shared,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()









    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")





