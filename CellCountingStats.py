"""
Author: Emilie Thelie (emilie.thelie@austin.utexas.edu)
UT Austin - July 2025

Python script to compute cell counting statistics for filaments.
Input:
    filename_filaments: str
        Path and name for the pkl file containing the filaments.
    filename_output: str
        Path and name for the output file. The ".npz" extension is directly added in the code and thus not needed in this argument.
    Lbox: int
        Physical size of the simulation box from which the filaments come from. Can be given in cMpc or cMpc/h. 
        All the lengths in this script will have the same units as Lbox.
    Nbins_cellCountingLen: int
        Number of bins in length for the PDF of lengthes.
    Nbins_lenFraction: int
        Number of bins for the length fraction vs cell fraction statistics.
    Nbins_cellCountingNcp: int
        Number of bins in critical point number for the PDF.
    Nbins_cpFraction: int
        Number of bins for the critical point number fraction vs cell fraction statistics.
    grid_res: int
        Resolution of the cells. Multiple values can be given in the arguments.
    
This code generates a npz file containing:
- Lengths PDF (units of 1/Lbox units) vs length bins (Lbox units) in cubic subvolumes with different resolution cube sizes,
- Length fraction (percentage) vs cell fraction (percentage) with different resolution cube sizes.

Example of a command line to run this script:
    $ python CellCountingStats.py ./data/filament_path.pkl ./data/CellCountingStats 111 20 30 20 20 1 2 4 8 10
"""

import sys
import pickle
import numpy as np

import time

##### To read input arguments
def read_input_args():
    filepath_filaments = sys.argv[1]
    print(f"    > Filename containing the filaments data: {filepath_filaments}.")
    
    filename_output = sys.argv[2]+".npz"
    print(f"    > Filename for the output data: {filename_output}.")
    
    Lbox = int(sys.argv[3])
    print(f"    > Physical size of the simulation box: {Lbox} cMpc (or cMpc/h).")

    Nbins_cellCountingLen = int(sys.argv[4])
    print(f"    > Number of bins in length for the PDF of lengthes: {Nbins_cellCountingLen}.")

    Nbins_lenFraction = int(sys.argv[5])
    print(f"    > Number of bins for the length fraction vs cell fraction statistics: {Nbins_lenFraction}.")

    Nbins_cellCountingNcp = int(sys.argv[6])
    print(f"    > Number of bins in critical point number for the PDF: {Nbins_cellCountingNcp}.")

    Nbins_cpFraction = int(sys.argv[7])
    print(f"    > Number of bins for the critical point number fraction vs cell fraction statistics: {Nbins_cpFraction}.")    
    
    grid_res = [int(arg) for arg in sys.argv[8:]]
    print(f"    > Coarse resolution of the cells: {grid_res} cMpc (or cMpc/h).")
    return filepath_filaments, filename_output, Lbox, Nbins_cellCountingLen, Nbins_lenFraction, Nbins_cellCountingNcp, Nbins_cpFraction, grid_res
    

##### To read filaments
def read_filaments_data(filepath_filaments):
    """
    Read the filaments from the pkl file, and return arrays of containing their information.
    Input:
        filepath_filaments: str
            Path to the pkl file containing the filaments. 
            As a note, a filament is defing by a set of points that delimit its segments.
    Output:
        id_fil: 1D array
            Id of the filament to which the points belong.
        crit_fil: 1D array
            Critical type index of the points. 
            -1 is a point that is not at the extremity of the filament, 1 and 2 are saddle points, and 3 is a maxima.
        pos_fil: 2D array
            3D position of the points along the filaments.
    """
    # reading filaments
    with open(filepath_filaments, "rb") as f:
        filament_path = pickle.load(f)
    
    # converting filament data into arrays
    id_fil, crit_fil, pos_fil = [], [], []
    for n_fil in range(len(filament_path)):
        for i_ptonfil in range(len(filament_path[n_fil])):
            id_fil.append(n_fil)
            crit_fil.append(filament_path[n_fil][i_ptonfil][0])
            pos_fil.append(filament_path[n_fil][i_ptonfil][1])
    id_fil = np.array(id_fil)
    crit_fil = np.array(crit_fil)
    pos_fil = np.array(pos_fil)
    Nfil = len(np.unique(id_fil))
    print(f"    > There are {Nfil} filaments and {len(id_fil)} filament segments.")
    print(f"    > There are {np.unique(pos_fil[(np.where(crit_fil==2)[0])],axis=0).shape[0]} critical points of index 2.")
    print(f"    > There are {np.unique(pos_fil[(np.where(crit_fil==3)[0])],axis=0).shape[0]} critical points of index 3 (maxima).")
    return id_fil, crit_fil, pos_fil


##### To compute length and midpoint of filament segments
def compute_filseg_length(pos_onefil):
    """
    Computes the length of all segments of the filament.
    Input:
        pos_onefil: 2D array
            Position of the points along the filaments (two following points delimiting segments).
    Output: 
        2D array
            Length of all the filament segments.        
    """
    return np.sqrt(np.sum(np.diff(pos_onefil,axis=0)**2,axis=1))

def compute_fil_length_midpoint(pos_fil,id_fil):
    """
    Compute the length and midpoint of the filaments segments, as well as the filaments length.
    Input:
        pos_fil: 2D array
            3D position of all filaments points.
        id_fil: 1D array
            Id of the filament to which the points belong.
    Output:
        id_filseg: 1D array
            Id of the filament to which each segment belongs.
        pos_filseg_midpoint: 2D array
            3D position of the midpoint of all filaments segments.
        filseg_length: 1D array
            Length of all filaments segments.
        fil_length: 1D array
            Length of all filaments.
    """
    Nfil = len(np.unique(id_fil))

    id_filseg = np.zeros(id_fil.shape[0]-Nfil,dtype=int)
    filseg_length = np.zeros(id_fil.shape[0]-Nfil)
    pos_filseg_midpoint = np.zeros((id_fil.shape[0]-Nfil,3))
    fil_length = np.zeros(Nfil)
    for n_fil in range(Nfil):
        # retrieve the filament
        id_fil_n = np.where(id_fil==n_fil)[0]
        pos_fil_n = pos_fil[id_fil_n]
        id_fil_n_segmidpoints = id_fil_n[:-1]-n_fil
        # save the id of the current filament for all the segments
        id_filseg[id_fil_n_segmidpoints] = n_fil
        # compute the length of its segments
        fil_n_length = compute_filseg_length(pos_fil_n)
        filseg_length[id_fil_n_segmidpoints] = fil_n_length
        # compute the mid point 
        pos_filseg_midpoint[id_fil_n_segmidpoints] = (pos_fil_n[1:]+pos_fil_n[:-1])/2
        # compute the length of the filaments
        fil_length[n_fil] = np.sum(fil_n_length)
    
    Lfil = np.sum(fil_length)
    print(f"    > Total length of filaments: {Lfil:.2f} cMpc (or cMpc/h depending on Lbox units).")

    return id_filseg, pos_filseg_midpoint, filseg_length, fil_length


##### To generate grids
def generate_fillength_grid(Lbox,pos_filseg_midpoint,filseg_length):
    """
    Generate a grid of resolution 1 cMpc (or cMpc/h according to Lbox) that contains the length of all filament falling into the cells.
    Filament are flagged as within a cell if their midpoint belongs to this cell.
    Input:
        Lbox: int
            Physical size of the simulation box from which the filaments come from.
        pos_filseg_midpoint: 2D array
            3D position of the midpoint of all filaments segments.
        filseg_length: 1D array
            Length of all filaments segments.
    Output: 3D array
        Grid containing the length of all filament falling into the cells.
    """
    pos_filseg_midpoint_in_cell = np.int64(np.floor(pos_filseg_midpoint))
    
    length_grid = np.zeros((Lbox,Lbox,Lbox))
    for i_bit in range(pos_filseg_midpoint_in_cell.shape[0]):
        length_grid[tuple(pos_filseg_midpoint_in_cell[i_bit])] += filseg_length[i_bit]

    print(f"    > Check of the total length of filaments inside the grid with dx=1: {np.sum(length_grid):.2f} cMpc or cMpc/h.")
    return length_grid

def generate_critptcount_grid(Ncells,which_crittype,crit_fil,pos_fil):
    """
    Generate a grid of resolution 1 cMpc (or cMpc/h according to Lbox) that contains the number of critical points of a given critical index.
    Input:
        Lbox: int
            Physical size of the simulation box from which the filaments come from.
        which_crittype: int
            Critical index of critical points, should be 0 (minima), 1, 2, or 3 (maxima) for 3D data.
        crit_fil: 1D array
            Critical type index of the points. 
        pos_fil: 2D array
            3D position of all filaments points.
    Output: 3D array
        Grid containing the length of all filament falling into the cells.
    """
    pos_critpt = np.unique(pos_fil[(np.where(crit_fil==which_crittype)[0])],axis=0)
    pos_critpt_in_cell = np.int64(np.floor(pos_critpt))
    

    critpt_grid = np.zeros((Ncells,Ncells,Ncells))
    for i_bit in range(pos_critpt_in_cell.shape[0]):
        critpt_grid[tuple(pos_critpt_in_cell[i_bit])] += 1

    return critpt_grid

def block_sum(grid, dx):
    """
    Perform a summation of cells values over cubic kernels of a given resolution.
    Input:
        grid: 3D array
            Input grid.
        dx: int
            Resolution of the cubic kernel.
    Output: 3D array
        Output reduced grid, which will have Ninput/dx cells on a side, with Ninput the number of cells on a side of the input grid.
    """
    # Calculate the trimmed shape
    trimmed_shape = [dim - (dim % dx) for dim in grid.shape] # in case the initial box size is not a multiple of the reduced box size
    trimmed = grid[tuple(slice(0, ts) for ts in trimmed_shape)]
    # Reshape into blocks
    reshaped = trimmed.reshape(trimmed_shape[0] // dx, dx,
                               trimmed_shape[1] // dx, dx,
                               trimmed_shape[2] // dx, dx)
    # Sum across the block dimensions
    reduced = reshaped.sum(axis=(1, 3, 5))
    return reduced


##### To compute cell counting statistics
def compute_cellCountingStat(grids,Nbins):
    """
    Compute the PDF of the gridded quantity within the cells.
    Input:
        grids: list
            3D grids of some quantity (e.g. filament length or number of critical points) for the different cell resolution.
        Nbins: int
            Number of bins for the PDF.
    Output:
        bins: 1D array
            Bins of the PDF (in units of the gridded quantity).
        cellCountingStat: 2D array
            PDF of the gridded quantity within the cells (in units of the gridded quantity), for every cell resolution.
    """
    bins_cellCountingLen = np.linspace(1e-5,20,Nbins)
    bins_len = bins_cellCountingLen[:-1]*0.5+bins_cellCountingLen[1:]*0.5

    cellCountingLen = np.zeros((len(grids),len(bins_len)))
    for i in range(len(grids)):
        cellCountingLen[i], _ = np.histogram(grids[i],bins_cellCountingLen,density=True)

    return bins_len, cellCountingLen

def compute_quantity_fraction(grids, Nbins):
    """
    Compute the quantity fraction with respect to the cell fraction.
    Input:
        grids: list
            3D grids of some quantity (e.g. filament length or number of critical points) for the different cell resolution.
        Nbins: int
            Number of bins in cell fraction.
    Output:
        cell_fraction: 1D array
            Cell percentage.
        qty_fraction: 2D array
            Percentage of the quantity there is in a given percentage of cells, for every cell resolution.
    """
    cell_fraction = np.linspace(0,100,Nbins)

    qty_fraction = np.zeros((len(grids),Nbins))
    for i in range(len(grids)):
        for j in range(len(cell_fraction)):
            grid = grids[i]
            total = np.sum(grid)
            qty_fraction[i,j] = np.sum(grid[ grid >= np.percentile(grid,100-cell_fraction[j]) ]) / total * 100

    return cell_fraction, qty_fraction



##### Running the script
if __name__ == "__main__":
    global_start_time = time.time()
    
    print("***** Reading parameters *****")
    filepath_filaments, filename_output, Lbox, Nbins_cellCountingLen, Nbins_lenFraction, Nbins_cellCountingNcp, Nbins_cpFraction, grid_res = read_input_args()

    print("\n***** Reading filaments *****")
    id_fil, crit_fil, pos_fil = read_filaments_data(filepath_filaments)

    print("\n***** Computing length and midpoint of filament segments *****")
    id_filseg, pos_filseg_midpoint, filseg_length, fil_length = compute_fil_length_midpoint(pos_fil,id_fil)

    print("\n***** Generating filament length grids *****")
    grid_fillength = generate_fillength_grid(Lbox,pos_filseg_midpoint,filseg_length)
    grid_fillength_allres = []
    for dx in grid_res:
        if dx==1:
            grid_fillength_allres.append(grid_fillength)
        else:
            grid_fillength_allres.append(block_sum(grid_fillength,dx))
            print(f"    > Check of the total length of filaments inside the grid with dx={dx}: {np.sum(grid_fillength_allres[-1]):.2f} cMpc (or cMpc/h).")

    print("\n***** Generating critical point number grids *****")
    grid_cp2, grid_cp3 = generate_critptcount_grid(Lbox,2,crit_fil,pos_fil), generate_critptcount_grid(Lbox,3,crit_fil,pos_fil)
    grid_cp2_allres, grid_cp3_allres = [], []
    for dx in grid_res:
        if dx==1:
            grid_cp2_allres.append(grid_cp2)
            grid_cp3_allres.append(grid_cp3)
        else:
            grid_cp2_allres.append(block_sum(grid_cp2,dx))
            print(f"    > Check of the total number of critical point of index 2 inside the grid with dx={dx}: {np.sum(grid_cp2_allres[-1]):.2f}.")
            grid_cp3_allres.append(block_sum(grid_cp3,dx))
            print(f"    > Check of the total number of critical point of index 3 inside the grid with dx={dx}: {np.sum(grid_cp3_allres[-1]):.2f}.")

    print("\n***** Computing and saving cell counting statistics *****")
    bins_len, cellCountingLen = compute_cellCountingStat(grid_fillength_allres,Nbins_cellCountingLen)
    cell_fraction, length_fraction_allres = compute_quantity_fraction(grid_fillength_allres, Nbins_lenFraction)
    
    bins_cp2, cellCountingcp2 = compute_cellCountingStat(grid_cp2_allres,Nbins_cellCountingNcp)
    bins_cp3, cellCountingcp3 = compute_cellCountingStat(grid_cp3_allres,Nbins_cellCountingNcp)
    cell_fraction_cp2, cp2_fraction_allres = compute_quantity_fraction(grid_cp2_allres, Nbins_cpFraction)
    cell_fraction_cp3, cp3_fraction_allres = compute_quantity_fraction(grid_cp3_allres, Nbins_cpFraction)

    np.savez(filename_output,
             bins_len=bins_len,cellCountingLen=cellCountingLen,cell_fraction=cell_fraction,length_fraction_allres=length_fraction_allres,
             bins_cp2=bins_cp2,cellCountingcp2=cellCountingcp2,cell_fraction_cp2=cell_fraction_cp2,cp2_fraction_allres=cp2_fraction_allres,
             bins_cp3=bins_cp3,cellCountingcp3=cellCountingcp3,cell_fraction_cp3=cell_fraction_cp3,cp3_fraction_allres=cp3_fraction_allres)
    
    print(f"\n***** Total run time: {time.time()-global_start_time:.2f}s *****")











