
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
    if processor == "processor0":
        print(cell_list)
        print(face_list)
        print(face_list[cell_list == 0])




    Link_Wise_Lattice = pd.DataFrame(
        {
            "cell_no":cell_list,
            "face": face_list,
        }
    )



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
