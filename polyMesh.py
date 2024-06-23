
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
import pprint
import pandas as pd

def read_polyMesh(processor, base_size, shift_shared, maxCellLevel_shared, lock_shared):
    points = read_PointsFile(processor)
    faces = read_FacesFile(processor)
    owner = read_CellFaceFile(processor, "owner")
    neighbour = read_CellFaceFile(processor, "neighbour")
    cellLevel = read_CellLevelFile(processor)
    boundaries = read_BoundaryFile(processor)

    min_SubdomainRange, max_SubdomainRange = read_Range_of_Subdomains(points)

    lock_shared.acquire()
    try:
        if processor == "processor0":
            x, y, z = min_SubdomainRange
            shift_shared = (x, y, z)
    finally:
        lock_shared.release()
    shift_shared = np.array(shift_shared)

    lock_shared.acquire()
    try:
        tmp = max(cellLevel)
        maxCellLevel_shared.value = max([maxCellLevel_shared.value, tmp])
    finally:
        lock_shared.release()
    max_cellLevel = maxCellLevel_shared.value

    normalized_Length = (2 ** (max_cellLevel + 1)) / base_size

    p = np.array(points)
    p = (p - shift_shared) * normalized_Length
    normalized_points = {i: np.array(
                         (int(Decimal(v[0]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                          int(Decimal(v[1]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                          int(Decimal(v[2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))), \
                          dtype = np.int32) for i, v in enumerate(p)}



    owner_face = np.arange(len(owner), dtype = np.int32)
    neighbour_face = np.arange(len(neighbour), dtype = np.int32)
    face_list = np.append(owner_face, neighbour_face)

    owner_cell = np.array(owner, dtype = np.int32)
    neighbour_cell = np.array(neighbour, dtype = np.int32)
    cell_list = np.append(owner_cell, neighbour_cell)

    polyMesh_df = pd.DataFrame(
        {
            "cell": cell_list,
            "face": face_list,
        }
    )



    unique_face_list = np.unique(face_list)
    unique_face_point_dict = {n: np.array([normalized_points[p] for p in faces[n]]) for n in unique_face_list}
    unique_face_minmax_dict = {n: np.append(np.min(p, axis = 0), np.max(p, axis = 0)) for n, p in unique_face_point_dict.items()}
    face_minmax_list = np.array([unique_face_minmax_dict[n] for n in face_list])

    polyMesh_df["fxmin"] = face_minmax_list[:, 0]
    polyMesh_df["fymin"] = face_minmax_list[:, 1]
    polyMesh_df["fzmin"] = face_minmax_list[:, 2]
    polyMesh_df["fxmax"] = face_minmax_list[:, 3]
    polyMesh_df["fymax"] = face_minmax_list[:, 4]
    polyMesh_df["fzmax"] = face_minmax_list[:, 5]



    pmin_df = polyMesh_df[["cell", "fxmin", "fymin", "fzmin"]].groupby("cell").min()
    pmax_df = polyMesh_df[["cell", "fxmax", "fymax", "fzmax"]].groupby("cell").max()

    unique_cell_index = np.array(pmin_df.index, dtype = np.int32)
    pmin = np.array(pmin_df)
    pmax = np.array(pmax_df)
    pmean = np.array([pmin, pmax]).mean(axis = 0, dtype = np.int32)
    size = (pmax - pmin)[:, 0]

    polyMesh_df["size"] = np.array([size[n] for n in cell_list])
    polyMesh_df["nx"] = np.array([pmean[n] for n in cell_list])[:, 0]
    polyMesh_df["ny"] = np.array([pmean[n] for n in cell_list])[:, 1]
    polyMesh_df["nz"] = np.array([pmean[n] for n in cell_list])[:, 2]



    def fcsys(pn, fmin, fmax):
        bl = fmin == fmax
        nc = pn[bl][0]
        nf = fmin[bl][0]
        sys = np.array([[1, 3], [2, 4], [5, 6]])
        return sys[bl][0][[nc < nf, nf < nc]][0]

    pn = np.array(polyMesh_df[["nx", "ny", "nz"]])
    fmin = np.array(polyMesh_df[["fxmin", "fymin", "fzmin"]])
    fmax = np.array(polyMesh_df[["fxmax", "fymax", "fzmax"]])

    fcsys_list = np.array([fcsys(p, fm, fx) for p, fm, fx in zip(pn, fmin, fmax)], dtype = np.int32)
    polyMesh_df["fcsys"] = fcsys_list



    nInternalFaces = len(neighbour)
    nFaces = np.array([v["nFaces"] for v in boundaries], dtype = np.int32)
    startFace = np.array([v["startFace"] for v in boundaries], dtype = np.int32)
    endFace = startFace + nFaces
    boundaryType = [v["type"] for v in boundaries]
    boundaryName = [v["name"] for v in boundaries]
    def face_condition(faceNo):
        if faceNo < nInternalFaces:
            return ["InternalFace", "None"]
        k = np.argmax(np.all([startFace <= faceNo, faceNo < endFace], axis = 0))
        return [boundaryType[k], boundaryName[k]]

    face_condition_list = np.array(list(map(face_condition, face_list)))
    polyMesh_df["boundaryType"] = face_condition_list[:, 0]
    polyMesh_df["boundaryName"] = face_condition_list[:, 1]





    amp = size.reshape([len(size), 1])
    pcsys_dict = {n: pmean + (amp * np.array(v)) for n, v in csys.items()}
    tmp_list = np.concatenate(list(pcsys_dict.values()))
    tmp_list = np.concatenate([pmean, tmp_list])
    tmp_list = np.array(pd.DataFrame(data = tmp_list).drop_duplicates())
    tmp_list = np.concatenate([pmean, tmp_list])
    ghostCell_df = pd.DataFrame(data = tmp_list).drop_duplicates(keep = False)
    ghostCell = np.array(ghostCell_df)

    ns = np.max(unique_cell_index) + 1
    ne = ns + len(ghostCell_df)
    unique_ghost_index = np.arange(ns, ne, dtype = np.int32)

    tmp_cell = np.concatenate([pmean, ghostCell])
    tmp_index = np.append(unique_cell_index, unique_ghost_index)
    pinv = {tuple(v): n for n, v in zip(tmp_index, tmp_cell)}
    pcsys_tuple = tuple(map(tuple, np.concatenate([pcsys_dict[k] for k in sorted(list(csys.keys()))])))

    ncell = len(unique_cell_index)
    csys_cell_index = np.array([pinv[v] for v in pcsys_tuple], dtype = np.int32).reshape([ncell, -1], order = "F")
    csys_cell_df = pd.DataFrame(csys_cell_index).add_prefix("csys_")
    csys_cell_df["cell"] = unique_cell_index


    Link_Wise_Lattice = pd.DataFrame(
        {
            "cell": unique_cell_index,
            "size": size,
            "nx": pmean[:,0],
            "ny": pmean[:,1],
            "nz": pmean[:,2],
        }
    )

    Link_Wise_Lattice = Link_Wise_Lattice.merge(csys_cell_df)





if __name__ == '__main__':
    from multiprocessing import Process, Manager, Array, Value, Lock
    import time

    base_size = 1.0
    print()
    # print(f"Importing OpenFOAM polyMesh ...", end=" ")
    print(f"Importing OpenFOAM polyMesh ...")
    start = time.perf_counter()
    array = Array("f", 3)
    value = Value("i", 0)
    manager = Manager()
    pmesh_dict = manager.dict()
    lock = Lock()
    process_list = list()
    for i in range(4):
        processor = "processor" + str(i)
        process = Process(
            target = read_polyMesh,
            kwargs = {
                "processor": processor,
                "base_size": base_size,
                "shift_shared": array,
                "maxCellLevel_shared": value,
                "lock_shared": lock,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")







    # read_polyMesh("processor0")
