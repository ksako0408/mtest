import polyMesh
import numpy as np
import os
import time
import pprint
import re
from operator import itemgetter
from multiprocessing import Process, Manager, Array, Value, Lock

# import networkx as nx
# import matplotlib.pyplot as plt
# import decimal
# from decimal import Decimal, ROUND_HALF_UP


# ---------------------------------------------------------------------------------------------------------------

base_size = 1.0



# ---------------------------------------------------------------------------------------------------------------
def import_OpenFOAMBoxelMesh(processor: str, shift: float, maxCellLevel: int, pmesh_dict: dict, lock: object):
    pMesh = polyMesh.polymesh(processor, base_size, shift, maxCellLevel, lock)
    pmesh_dict[processor] = pMesh


def const_LatticeFromPolyMesh(processor: str, lattice_dict: dict, pmesh_dict: dict):
    pMesh = pmesh_dict[processor]
    mesh = polyMesh.const_FluidMesh(pMesh)
    lattice_dict[processor] = mesh



# --------------------------------------------------------------------------

def get_number_of_Subdomains():
    procNo = 0
    while os.path.isdir("processor" + str(procNo)):
        procNo += 1
    return procNo


def get_Range_of_Subdomains(processor, range_of_subdomains):
    # print(f'processor: {processor} started.')
    points = polyMesh.read_PointsFile(processor)
    p = np.array(points)
    p_xmin = p[:, 0].min()
    p_ymin = p[:, 1].min()
    p_zmin = p[:, 2].min()
    p_min = (p_xmin, p_ymin, p_zmin)

    p_xmax = p[:, 0].max()
    p_ymax = p[:, 1].max()
    p_zmax = p[:, 2].max()
    p_max = (p_xmax, p_ymax, p_zmax)
    # print(f'processor: {processor} ended.')

    range_of_subdomains[processor] = (p_min, p_max)

def find_ReferencedCell_of_Processors(processor: str, lattice_dict: dict, pmesh_dict: dict):
    mesh = lattice_dict[processor]
    other_mesh = {proc: lattice_dict[proc] for proc in lattice_dict.keys() if not proc == processor}
    range_dict = {proc: pmesh_dict[proc].nrml_domainRange for proc in pmesh_dict.keys()}
    mesh = polyMesh.update_boundaryType(processor, mesh, other_mesh, range_dict)
    lattice_dict[processor] = mesh


def reconstruct_lattice(processor, lattice_dict, mesh_dict, tables_dict):
    mesh = lattice_dict[processor]
    table = tables_dict[processor]
    other_table = {proc: t for proc, t in tables_dict.items() if not proc == processor}
    ret = polyMesh.generate_GhostCell(processor, mesh, table, other_table)
    mesh_dict[processor] = ret


def translate_CellCenter(processor, lattice_dict, tables_dict):
    mesh = lattice_dict[processor]
    table = polyMesh.tables_CellCenter_to_CellNo(processor, mesh)
    tables_dict[processor] = table


# --------------------------------------------------------------------------

if __name__ == '__main__':

    processor_Num = get_number_of_Subdomains()
    print(f"Number of Subdomains: {processor_Num}")


    print()
    print(f"Importing OpenFOAM polyMesh ...", end=" ")
    start = time.perf_counter()
    array = Array("f", 3)
    value = Value("i", 0)
    manager = Manager()
    pmesh_dict = manager.dict()
    lock = Lock()
    process_list = list()
    for i in range(processor_Num):
        proc = "processor" + str(i)
        process = Process(
            target = import_OpenFOAMBoxelMesh,
            kwargs = {
                "processor": proc,
                "shift": array,
                "maxCellLevel": value,
                "pmesh_dict": pmesh_dict,
                "lock": lock,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")

    print(f"Range of Subdomains (p_min, pmax)")
    for i in range(processor_Num):
        processor = "processor" + str(i)
        print(f'  processor{i}: {pmesh_dict[processor].domainRange}')



    print()
    print(f"Constructing Link-Wise Lattice from OpenFOAM polyMesh ...", end=" ")
    start = time.perf_counter()
    lattice_dict = manager.dict()
    process_list = list()
    for i in range(processor_Num):
        proc = "processor" + str(i)
        process = Process(
            target = const_LatticeFromPolyMesh,
            kwargs = {
                "processor": proc,
                "lattice_dict": lattice_dict,
                "pmesh_dict": pmesh_dict,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")



    print()
    print(f"Finding Reference Node of Each Processors ...", end=" ")
    start = time.perf_counter()
    process_list = list()
    for i in range(processor_Num):
        proc = "processor" + str(i)
        process = Process(
            target = find_ReferencedCell_of_Processors,
            kwargs = {
                "processor": proc,
                "lattice_dict": lattice_dict,
                "pmesh_dict": pmesh_dict,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")

    # for i in range(processor_Num):
    #     proc = "processor" + str(i)
    #     mesh = lattice_dict[proc]
    #     for c in mesh:
    #         boundaryType = mesh[c].boundaryType
    #         boundaryName = mesh[c].boundaryName
    #         tmpDict = {k: boundaryName[k] for k, v in boundaryType.items() if v == "processor"}
    #         if not tmpDict: continue
    #         print(f"\nprocessor{i}: cell={c}")
    #         for k in mesh[c].boundaryType:
    #             print(f"{k}: {mesh[c].boundaryType[k]}, {mesh[c].boundaryName[k]}")


    print()
    print(f"Generating Translate Tables CellCenter to CellNo ...")
    print()
    start = time.perf_counter()
    tables_dict = manager.dict()
    for i in range(processor_Num):
        proc = "processor" + str(i)
        process = Process(
            target = translate_CellCenter,
            kwargs = {
                "processor": proc,
                "lattice_dict": lattice_dict,
                "tables_dict": tables_dict,
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")


    print()
    # print(f"Generating Ghost Cell ...", end=" ")
    print(f"Generating Ghost Cell ...")
    print()
    start = time.perf_counter()
    mesh_dict = manager.dict()
    process_list = list()
    for i in range(processor_Num):
        proc = "processor" + str(i)
        process = Process(
            target = reconstruct_lattice,
            kwargs = {
                "processor": proc,
                "lattice_dict": lattice_dict,
                "mesh_dict": mesh_dict,
                "tables_dict": tables_dict
            }
        )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    end = time.perf_counter()
    print(f"Done.\n (elapsed Time: {end - start} sec.)")
    exit()


    # pprint.pprint(len(lattice_dict["processor0"]))
    # pprint.pprint(mesh_dict["processor0"][0].size)
    # pprint.pprint(mesh_dict["processor0"][0].center)
    # pprint.pprint(mesh_dict["processor0"][0].neighbourCell)


    exit()







    # print(f"Range of Subdomains (p_min, pmax)")
    # manager = Manager()
    # minPoint = Array("f", 3)
    # range_of_subdomains = manager.dict()
    # process_list = list()
    # for i in range(processor_Num):
    #     proc = "processor" + str(i)
    #     process = Process(
    #         target = get_Range_of_Subdomains,
    #         kwargs = {
    #             "processor": proc,
    #             "range_of_subdomains": range_of_subdomains,
    #         }
    #     )
    #     process.start()
    #     process_list.append(process)

    # for process in process_list:
    #     process.join()

    # for i in range(processor_Num):
    #     processor = "processor" + str(i)
    #     print(f'  processor{i}: {range_of_subdomains[processor]}')




    # print()
    # print(f"Importing OpenFOAM polyMesh ...", end=" ")
    # start = time.perf_counter()
    # shift = range_of_subdomains["processor0"][0]
    # manager = Manager()
    # mesh_dict = manager.dict()
    # process_list = list()
    # for i in range(processor_Num):
    #     proc = "processor" + str(i)
    #     process = Process(
    #         target = import_OpenFOAMBoxelMesh,
    #         kwargs = {
    #             "processor": proc,
    #             "shift": shift,
    #             "mesh_dict": mesh_dict,
    #         }
    #     )
    #     process.start()
    #     process_list.append(process)

    # for process in process_list:
    #     process.join()

    # end = time.perf_counter()
    # print(f"Done. (elapsed Time: {end - start} sec.)")


    # for proc in mesh_dict:
    #     mesh = mesh_dict[proc]["mesh"]
    #     cid =[mesh[c].id for c in mesh]
    #     print(proc, max(cid))



    # print()
    # print(f"Finding ReferenceCell of Each Processors ...", end=" ")
    # start = time.perf_counter()

    # process_list = list()
    # for i in range(processor_Num):
    #     proc = "processor" + str(i)
    #     process = Process(
    #         target = find_ReferencedCell_of_Processors,
    #         kwargs = {
    #             "processor": proc,
    #             "mesh_dict": mesh_dict,
    #             "range_dict": range_of_subdomains,
    #         }
    #     )
    #     process.start()
    #     process_list.append(process)

    # for process in process_list:
    #     process.join()

    # end = time.perf_counter()
    # print(f"Done. (elapsed Time: {end - start} sec.)")


    # mesh_dict["processor1"] = "test"
    # for i in mesh_dict:
    #     print(i)
    #     d = mesh_dict[i]
    #     print(type(d))





# Processor 0
#     Number of cells = 1730
#     Number of faces shared with processor 1 = 51
#     Number of faces shared with processor 2 = 56
#     Number of processor patches = 2
#     Number of processor faces = 107
#     Number of boundary faces = 1781






    # constMeshNetwork(proc, 3)

    # proc = "processor" + str(1)
    # constMeshNetwork(proc, 2)
    # proc = "processor" + str(2)
    # constMeshNetwork(proc, 2)
    # proc = "processor" + str(3)
    # constMeshNetwork(proc, 2)


    # exit()
    # nrml_points, characteristic_length, max_cellLevel = mr.get_normalizedPoints(processor, base_size)
    # points_in_face = mr.read_FacesFile(processor)
    # faces_in_cell, owner_cell = mr.read_CellsFile(processor)
    # boundary_faces = mr.read_BoundaryFile(processor)



    # points_in_cell = dict()
    # for cell, faces in faces_in_cell.items():
    #     tmp = set()
    #     for face in faces:
    #         tmp.update(set(points_in_face[face]))
    #     points_in_cell[cell] = tmp


    # vertex_in_cell = dict()
    # cell_center = dict()
    # cell_levels = dict()
    # cell_size = dict()

    # for k, points in points_in_cell.items():
    #     p_max = np.max(np.array([nrml_points[p] for p in points]), axis = 0)
    #     p_min = np.min(np.array([nrml_points[p] for p in points]), axis = 0)
    #     pc = np.mean([p_max, p_min], axis = 0, dtype = np.int32)

    #     vertex = {p: csys_v[tuple(np.sign(np.array(nrml_points[p]) - pc))] for p in points}
    #     vertex_in_cell[k] = vertex
    #     cell_center[k] = pc
    #     # cell_center[pc] = k

    #     dp = (p_max - p_min)[0]
    #     cell_size[k] = dp
    #     cell_level = max_cellLevel - (~dp & (dp -1)).bit_count() + 1
    #     cell_levels[k] = cell_level


    # cell_sorted_by_levels = sorted(cell_levels.items(), key=lambda x: x[1])
    # table_of_cell_sorted = {cell[0]: i for i, cell in enumerate(cell_sorted_by_levels)}


    # neighbor_cell = dict()

    # for boundary in boundary_faces:
    #     nFaces = boundary["nFaces"]
    #     startFace = boundary["startFace"]
    #     boundary_type = boundary["type"]
    #     boundary_name = boundary["name"]
    #     for n in range(startFace, startFace + nFaces):
    #         owner = owner_cell[n]
    #         points = points_in_face[n]
    #         vertex = {csys_v[tuple(np.sign(nrml_points[p] - cell_center[owner]))] for p in points}
    #         for i, vset in faces_def.items():
    #             if vertex.issubset(vset):
    #                 neighbor_cell[tuple(cell_center[owner])] = {i: boundary_type}

    # exit()

    # point_contact_to_cell = {cell: set() for cell in points_in_cell.keys()}
    # edge_contact_to_cell = {cell: set() for cell in points_in_cell.keys()}
    # face_contact_to_cell = {cell: set() for cell in points_in_cell.keys()}

    # for c1, c2 in itertools.combinations(points_in_cell.keys(), 2):
    #     p1 = points_in_cell[c1]
    #     p2 = points_in_cell[c2]
    #     p = p1.intersection(p2)

    #     if not p: continue

    #     n = len(p)
    #     if n == 1:
    #         point_contact_to_cell[c1].add(c2)
    #         point_contact_to_cell[c2].add(c1)
    #     elif n <= 3:
    #         edge_contact_to_cell[c1].add(c2)
    #         edge_contact_to_cell[c2].add(c1)
    #     else:
    #         face_contact_to_cell[c1].add(c2)
    #         face_contact_to_cell[c2].add(c1)





    # neighbor_cell = {k: np.full(26, -1, dtype = np.int32) for k in points_in_cell.keys()}
    # ghost_vertex = {k: np.arange(1, 27, dtype = np.int32) for k in points_in_cell.keys()}




    # for c0, cells in face_contact_to_cell.items():
    #     if not cells: continue
    #     p0_set = points_in_cell[c0]

    #     for c1 in cells:
    #         p1_set = points_in_cell[c1]
    #         p_set = {vertex_in_cell[c0][p] for p in p0_set.intersection(p1_set)}

    #         for k, ref in ref_face.items():
    #             if ref.issubset(p_set):
    #                 neighbor_cell[c0][k-1] = c1
    #                 tmp = ghost_vertex[c0]
    #                 ghost_vertex[c0] = tmp[tmp != k]


    # for c0, cells in edge_contact_to_cell.items():
    #     if not cells: continue
    #     p0_set = points_in_cell[c0]

    #     for c1 in cells:
    #         p1_set = points_in_cell[c1]
    #         p_set = {vertex_in_cell[c0][p] for p in p0_set.intersection(p1_set)}

    #         for k, ref in ref_edge.items():
    #             if ref.issubset(p_set):
    #                 neighbor_cell[c0][k-1] = c1
    #                 tmp = ghost_vertex[c0]
    #                 ghost_vertex[c0] = tmp[tmp != k]


    # for c0, cells in point_contact_to_cell.items():
    #     if not cells: continue
    #     p0_set = points_in_cell[c0]

    #     for c1 in cells:
    #         p1_set = points_in_cell[c1]
    #         p_set = {vertex_in_cell[c0][p] for p in p0_set.intersection(p1_set)}

    #         for k, ref in ref_point.items():
    #             if ref.issubset(p_set) and cell_size[c0] == cell_size[c1]:
    #                 neighbor_cell[c0][k-1] = c1
    #                 tmp = ghost_vertex[c0]
    #                 ghost_vertex[c0] = tmp[tmp != k]




    # ghost_point = dict()
    # for c0, vertex in ghost_vertex.items():
    #     for v in vertex:
    #         gp = tuple(np.array(csys[v]) * cell_size[c0] + cell_center[c0])
    #         iv = csys_inv[v]
    #         if gp not in ghost_point:
    #             ghost_point[gp] = {iv: c0}
    #         else:
    #             ghost_point[gp][iv] = c0




    # for boundary in boundary_faces:
    #     nFaces = boundary["nFaces"]
    #     startFace = boundary["startFace"]
    #     boundary_type = boundary["type"]
    #     boundary_name = boundary["name"]
    #     for n in range(startFace, startFace + nFaces):
    #         owner = owner_cell[n]
    #         points = points_in_face[n]
    #         vertex = {csys_v[tuple(np.sign(nrml_points[p] - cell_center[owner]))] for p in points}




    # ghost_set = set()
    # for c0, ghosts in ghost_cell.items():
    #     for v in ghosts:
    #         ghost_set.add(tuple(np.array(csys[v]) * cell_size[c0] + cell_center[c0]))
    # pprint.pprint(len(ghost_set))

    # ghost_cell_id = {v: k for k, v in enumerate(ghost_set)}
    # pprint.pprint(ghost_cell_id)
    # print(min(ghost_cell_id.values()))
    # print(max(ghost_cell_id.values()))
    # pprint.pprint(ghost_cell)

    # cmax = max(points_in_cell.keys())
    # print(cmax)
    # for c0, vertex in ghost_cell.items():
    #     for v in vertex:
    #         c1 = ghost_cell_id[tuple(np.array(csys[v]) * cell_size[c0] + cell_center[c0])]
    #         neighbor_cell[c0][v-1] = cmax + c1 + 1



    # pprint.pprint(ghost_point)
    # pprint.pprint(ghost_cell[790])
    # pprint.pprint(relative_level[790])
    # pprint.pprint(points_in_cell[79])
    # pprint.pprint(points_in_cell[790])
    # pprint.pprint(nrml_points[161])
    # pprint.pprint(nrml_points[163])
    # pprint.pprint(nrml_points[1032])
    # pprint.pprint(nrml_points[1034])
    # pprint.pprint(nrml_points[135])
    # pprint.pprint(nrml_points[136])
    # pprint.pprint(nrml_points[1006])
    # pprint.pprint(nrml_points[1007])
    # print("")
    # pprint.pprint(nrml_points[162])
    # pprint.pprint(nrml_points[164])
    # pprint.pprint(nrml_points[165])
    # pprint.pprint(nrml_points[1033])
    # pprint.pprint(nrml_points[1035])
    # pprint.pprint(nrml_points[1036])
    # pprint.pprint(cell_levels[79])
    # pprint.pprint(points_in_cell[79])
    # pprint.pprint([nrml_points[k] for k in points_in_cell[79]])
    # pprint.pprint(cell_levels[790])
    # pprint.pprint(points_in_cell[790])
    # pprint.pprint([nrml_points[k] for k in points_in_cell[790]])
    # pprint.pprint(cell_levels[28])
    # pprint.pprint(points_in_cell[28])
    # pprint.pprint([nrml_points[k] for k in points_in_cell[28]])

    # exit()





    # pmin = {k: np.array((min([nrml_points[p][0] for p in points]), \
    #                      min([nrml_points[p][1] for p in points]), \
    #                      min([nrml_points[p][2] for p in points]))) \
    #                          for k, points in points_in_cell.items()}
    # nz4 = np.zeros(4, dtype = np.int32)






    # ow = readCell(filePath + "owner")
    # ne = readCell(filePath + "neighbour")
    # cc, cs = getCellCenter(pt, fc, ow, ne)
    # cl = getCellLevel(cs)
    # print(pt)
    # all_cellCenter += cc
    # all_cellSize += cs
    # all_cellLevels += cl
# cl = readCellLevel(filePath + "cellLevel")
# print(cl)

    # max_inner_node = max(ow) + 1
    # pe = tuple(np.full(max_inner_node, subdomain))
    # lid = tuple(range(max_inner_node))

    # all_processors += pe
    # local_nodes += lid

    # return
