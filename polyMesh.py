
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

    normalized_points = {i: (int(Decimal(v[0]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                             int(Decimal(v[1]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), \
                             int(Decimal(v[2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP))) \
                             for i, v in enumerate(p)}

    owner_face = np.arange(len(owner), dtype = np.int32)
    neighbour_face = np.arange(len(neighbour), dtype = np.int32)
    face_list = np.append(owner_face, neighbour_face)

    owner_cell = np.array(owner, dtype = np.int32)
    neighbour_cell = np.array(neighbour, dtype = np.int32)
    cell_list = np.append(owner_cell, neighbour_cell)

    unique_cell_list = np.unique(cell_list)
    cell_face_list = [face_list[cell_list == n] for n in unique_cell_list]

    cell_size_list = list()
    cell_center_list = list()
    for cell_face in cell_face_list:
        px = set()
        py = set()
        pz = set()
        for f in cell_face:
            p_list = faces[f]
            for p in p_list:
                px.add(normalized_points[p][0])
                py.add(normalized_points[p][1])
                pz.add(normalized_points[p][2])
        xmin = min(px)
        ymin = min(py)
        zmin = min(pz)
        xmax = max(px)
        ymax = max(py)
        zmax = max(pz)
        cell_size_list.append(xmax - xmin)
        cell_center_list.append([int((xmax + xmin)/2), int((ymax + ymin)/2), int((zmax + zmin)/2)])

    cell_center_list = np.array(cell_center_list, dtype = np.int32)
    size_list = np.array([cell_size_list[n] for n in cell_list], dtype = np.int32)
    center_list = np.array([cell_center_list[n] for n in cell_list], dtype = np.int32)


    _ = [faces[n] for n in face_list]
    face_point_list = [np.array([normalized_points[p] for p in face_point]) for face_point in _]
    face_pminmax_list = [np.array([np.min(p, axis = 0), np.max(p, axis = 0)]) for p in face_point_list]


    def fcsys(n, f):
        f0 = face_pminmax_list[f][0]
        f1 = face_pminmax_list[f][1]
        bl = f0 == f1
        cn = cell_center_list[n][bl][0]
        fn = f0[bl][0]
        sys = np.array([[1, 3], [2, 4], [5, 6]])
        return sys[bl][0][np.array([cn < fn, fn < cn])][0]

    cell_face_order = [[fcsys(n, f) for f in cell_face] for n, cell_face in zip(unique_cell_list, cell_face_list)]

    if processor == "processor0":
        pprint.pprint(cell_face_order)
    #     pprint.pprint(cell_face_list[0])
    #     pprint.pprint(face_pminmax_list[0])
    #     pprint.pprint(cell_center_list[0])
    #     pprint.pprint(face_pminmax_list[0][0][face_pminmax_list[0][0] == face_pminmax_list[0][1]])
    #     pprint.pprint(cell_center_list[0][face_pminmax_list[0][0] == face_pminmax_list[0][1]])
    #     pprint.pprint(face_point_list[0])
        # pprint.pprint(cell_face_list)
        # pprint.pprint(face_list)
        # pprint.pprint(center_list)


    Link_Wise_Lattice = pd.DataFrame(
        {
            "cell_no":cell_list,
            "face": face_list,
            "size": size_list,
            "nx": center_list[:,0],
            "ny": center_list[:,1],
            "nz": center_list[:,2],
        }
    )




    # if processor == "processor0":
    #     print(Link_Wise_Lattice)



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
