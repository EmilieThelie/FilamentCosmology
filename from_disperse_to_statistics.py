import numpy as np
from datetime import datetime
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--segfile', type=str, required=True)
parser.add_argument('--critfile', type=str, required=True)
parser.add_argument('--galfile', type=str, default=None)
parser.add_argument('--outname', type=str, required=True)
parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if args.verbose:
    print('Starting..... ', formatted_time, flush=True)

##################################################################
################## Construct arcs and filaments ##################
##################################################################

################## Load DisPerSE output and pre##################

dtype = [('U0', 'f4'), ('U1', 'f4'), ('U2', 'f4'),
         ('V0', 'f4'), ('V1', 'f4'), ('V2', 'f4'),
         ('value_U', 'f4'), ('value_V', 'f4'),
         ('type', 'i4'), ('boundary', 'i4')]

segs = np.genfromtxt(args.segfile, dtype=dtype, comments='#', delimiter=None)

segs_Upos = np.vstack([segs['U0'], segs['U1'], segs['U2']]).T
segs_Vpos = np.vstack([segs['V0'], segs['V1'], segs['V2']]).T
segs_length = np.sqrt(np.sum((segs_Upos - segs_Vpos) ** 2, axis = 1))

dtype = [('X0', 'f4'), ('X1', 'f4'), ('X2', 'f4'),('value', 'f4'),
         ('type', 'i4'), ('pair_id', 'i4'), ('boundary', 'i4')]

crits = np.genfromtxt(args.critfile, dtype=dtype, comments='#')
mask = crits['type'] >= 2
crits = crits[mask]

crits_pos = np.vstack([crits['X0'], crits['X1'], crits['X2']]).T

segs_U_crit_type = np.zeros(len(segs), dtype = int) - 1
segs_V_crit_type = np.zeros(len(segs), dtype = int) - 1
segs_U_crit_idx = np.zeros(len(segs), dtype = int) - 1
segs_V_crit_idx = np.zeros(len(segs), dtype = int) - 1

def find_pos_in_arr(pos, arr, tolerance=1e-4):
    return np.where((np.abs(arr[:, 0] - pos[0]) < tolerance) & 
                    (np.abs(arr[:, 1] - pos[1]) < tolerance) & 
                    (np.abs(arr[:, 2] - pos[2]) < tolerance))[0]


for i, pos in enumerate(segs_Upos):
    try:
        segs_U_crit_idx[i] = find_pos_in_arr(pos, crits_pos)[0]
        segs_U_crit_type[i] = crits['type'][segs_U_crit_idx[i]]
    except:
        pass
for i, pos in enumerate(segs_Vpos):
    try:
        segs_V_crit_idx[i] = find_pos_in_arr(pos, crits_pos)[0]
        segs_V_crit_type[i] = crits['type'][segs_V_crit_idx[i]]
    except:
        pass

################## Optimized Arc Construction ##################

def create_arc_catalog_optimized(segs, crits):
    if args.verbose:
        print(f"Starting optimized processing with {len(segs)} segments and {len(crits)} critical points")
    
    # Pre-compute all unique positions and create mapping
    seg_u_pos = np.column_stack([segs['U0'], segs['U1'], segs['U2']])
    seg_v_pos = np.column_stack([segs['V0'], segs['V1'], segs['V2']])
    crit_pos = np.column_stack([crits['X0'], crits['X1'], crits['X2']])
    
    # Create all unique positions array for efficient lookup
    all_positions = np.vstack([seg_u_pos, seg_v_pos, crit_pos])
    unique_positions, inverse_indices = np.unique(all_positions, axis=0, return_inverse=True)
    
    n_segs = len(segs)
    n_crits = len(crits)
    
    # Map segment endpoints to position indices
    seg_u_pos_idx = inverse_indices[:n_segs]
    seg_v_pos_idx = inverse_indices[n_segs:2*n_segs]
    crit_pos_idx = inverse_indices[2*n_segs:2*n_segs+n_crits]
    
    # Create efficient lookup: position_idx -> critical_point_idx
    pos_to_crit = np.full(len(unique_positions), -1, dtype=int)
    pos_to_crit[crit_pos_idx] = np.arange(len(crits))
    
    # Create adjacency lists for graph traversal
    adjacency = defaultdict(list)
    for i in range(n_segs):
        u_idx = seg_u_pos_idx[i]
        v_idx = seg_v_pos_idx[i]
        adjacency[u_idx].append((v_idx, i))
        adjacency[v_idx].append((u_idx, i))
    
    if args.verbose:
        print(f"Created adjacency graph with {len(unique_positions)} nodes")
    
    # Track used segments
    used_segments = np.zeros(n_segs, dtype=bool)
    
    # Results
    arc_catalog = []
    partial_arcs = []
    
    direct_arcs = 0
    extended_arcs = 0
    dead_end_arcs = 0
    
    if args.verbose:
        print("Starting arc identification...")
    
    # Process all segments
    for seg_idx in range(n_segs):
        if seg_idx % 10000 == 0 and seg_idx > 0:
            if args.verbose:
                print(f"Processed {seg_idx}/{n_segs} segments. Found {len(arc_catalog)} complete arcs so far.")
        
        if used_segments[seg_idx]:
            continue
            
        u_pos_idx = seg_u_pos_idx[seg_idx]
        v_pos_idx = seg_v_pos_idx[seg_idx]
        
        u_crit_idx = pos_to_crit[u_pos_idx] if pos_to_crit[u_pos_idx] >= 0 else None
        v_crit_idx = pos_to_crit[v_pos_idx] if pos_to_crit[v_pos_idx] >= 0 else None
        
        # Case 1: Direct connection between two critical points
        if u_crit_idx is not None and v_crit_idx is not None:
            if crits[u_crit_idx]['type'] <= crits[v_crit_idx]['type']:
                arc_catalog.append((u_crit_idx, v_crit_idx, [seg_idx]))
            else:
                arc_catalog.append((v_crit_idx, u_crit_idx, [seg_idx]))
            used_segments[seg_idx] = True
            direct_arcs += 1
            continue
            
        # Case 2: One endpoint is a critical point - trace the arc
        if u_crit_idx is not None or v_crit_idx is not None:
            start_crit_idx = u_crit_idx if u_crit_idx is not None else v_crit_idx
            start_pos_idx = u_pos_idx if u_crit_idx is not None else v_pos_idx
            current_pos_idx = v_pos_idx if u_crit_idx is not None else u_pos_idx
            
            arc_segments = [seg_idx]
            used_segments[seg_idx] = True
            
            # Trace the arc
            while True:
                # Check if current position is a critical point
                end_crit_idx = pos_to_crit[current_pos_idx] if pos_to_crit[current_pos_idx] >= 0 else None
                if end_crit_idx is not None:
                    # Complete arc found
                    if crits[start_crit_idx]['type'] <= crits[end_crit_idx]['type']:
                        arc_catalog.append((start_crit_idx, end_crit_idx, arc_segments))
                    else:
                        arc_catalog.append((end_crit_idx, start_crit_idx, arc_segments[::-1]))
                    extended_arcs += 1
                    break
                
                # Find next unused segment
                next_seg_idx = None
                next_pos_idx = None
                
                for neighbor_pos_idx, neighbor_seg_idx in adjacency[current_pos_idx]:
                    if not used_segments[neighbor_seg_idx]:
                        next_seg_idx = neighbor_seg_idx
                        next_pos_idx = neighbor_pos_idx
                        break
                
                if next_seg_idx is None:
                    # Dead end
                    partial_arcs.append((start_crit_idx, arc_segments, current_pos_idx))
                    dead_end_arcs += 1
                    break
                
                # Continue arc
                arc_segments.append(next_seg_idx)
                used_segments[next_seg_idx] = True
                current_pos_idx = next_pos_idx
    
    # Count loose segments
    loose_segments = np.where(~used_segments)[0].tolist()
    
    if args.verbose:
        print("\nOptimized processing complete!")
        print(f"Direct arcs (single segment): {direct_arcs}")
        print(f"Extended arcs (multiple segments): {extended_arcs}")
        print(f"Partial arcs (dead ends): {dead_end_arcs}")
        print(f"Total complete arcs: {len(arc_catalog)}")
        print(f"Total partial arcs: {len(partial_arcs)}")
        print(f"Loose segments: {len(loose_segments)}")
    
    return arc_catalog, partial_arcs, loose_segments

# Create the catalog with optimized version
arc_catalog, partial_arcs, loose_segments = create_arc_catalog_optimized(segs, crits)

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if args.verbose:
    print('Arcs constructed..... ', formatted_time, flush=True)

################## Print summary ##################

# Print results
if args.verbose:
    print("\nSample complete arcs:")
    for i, (crit1, crit2, segments) in enumerate(arc_catalog[:5]):
        print(f"Arc {i}: Critical points {crit1} (type {crits[crit1]['type']}) -> {crit2} (type {crits[crit2]['type']})")
        print(f"  Segments: {segments}")
        print(f"  Arc length: {len(segments)} segments")
    
    print("\nSample partial arcs (dead ends):")
    for i, (crit_idx, segments, endpoint) in enumerate(partial_arcs[:5]):
        print(f"Partial arc {i}: From critical point {crit_idx} (type {crits[crit_idx]['type']}) to dead end at {endpoint}")
        print(f"  Segments: {segments}")
        print(f"  Arc length: {len(segments)} segments")
    
    print("\nSample loose segments:")
    for i in loose_segments[:5]:
        print(f"Segment {i}: ({segs[i]['U0']}, {segs[i]['U1']}, {segs[i]['U2']}) -> ({segs[i]['V0']}, {segs[i]['V1']}, {segs[i]['V2']})")
    
    # Calculate statistics on arc lengths
    if arc_catalog:
        complete_arc_lengths = [len(arc[2]) for arc in arc_catalog]
        print(f"\nComplete arc length statistics:")
        print(f"  Minimum: {min(complete_arc_lengths)} segments")
        print(f"  Maximum: {max(complete_arc_lengths)} segments")
        print(f"  Average: {sum(complete_arc_lengths)/len(complete_arc_lengths):.2f} segments")
    
    if partial_arcs:
        partial_arc_lengths = [len(arc[1]) for arc in partial_arcs]
        print(f"\nPartial arc length statistics:")
        print(f"  Minimum: {min(partial_arc_lengths)} segments")
        print(f"  Maximum: {max(partial_arc_lengths)} segments")
        print(f"  Average: {sum(partial_arc_lengths)/len(partial_arc_lengths):.2f} segments")

################## Save arc results ##################

import pickle

with open(args.outname + "_complete_arcs.pkl", "wb") as f:
    pickle.dump(arc_catalog, f)
with open(args.outname + "_partial_arcs.pkl", "wb") as f:
    pickle.dump(partial_arcs, f)
with open(args.outname + "_loose_segments.pkl", "wb") as f:
    pickle.dump(loose_segments, f)

################## Construct filaments from complete arcs ##################

from collections import defaultdict, deque
from itertools import combinations

def construct_filaments(arcs, crits):
    """
    Construct 2-3 filaments from arc catalog.
    
    Args:
        arcs: List of tuples (idx1, idx2, segments) where idx1 <= idx2
        crits: numpy array with dtype including 'type' field
    
    Returns:
        filament_arcs: List of filaments as lists of arc indices
        filament_segments: List of filaments as lists of segment indices
    """
    
    # Extract point types from crits array
    point_types = {i: crits[i]['type'] for i in range(len(crits))}
    
    # Build adjacency graph and arc lookup
    graph = defaultdict(set)
    arc_lookup = {}  # (pt1, pt2) -> arc_idx
    
    for i, (pt1, pt2, segs) in enumerate(arcs):
        graph[pt1].add(pt2)
        graph[pt2].add(pt1)
        arc_lookup[(min(pt1, pt2), max(pt1, pt2))] = i
    
    # Find all type-4 connected components (buckets)
    def get_connected_components(nodes_4):
        """Get connected components of type-4 nodes"""
        if not nodes_4:
            return []
        
        # Build subgraph of type-4 nodes only
        subgraph = defaultdict(set)
        for node in nodes_4:
            for neighbor in graph[node]:
                if neighbor in nodes_4:
                    subgraph[node].add(neighbor)
        
        # Find connected components
        visited = set()
        components = []
        
        for node in nodes_4:
            if node not in visited:
                component = set()
                queue = deque([node])
                
                while queue:
                    curr = queue.popleft()
                    if curr in visited:
                        continue
                    visited.add(curr)
                    component.add(curr)
                    
                    for neighbor in subgraph[curr]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    # Get all type-4 nodes and their connected components
    nodes_4 = {i for i, t in point_types.items() if t == 4}
    type4_components = get_connected_components(nodes_4)
    
    # Initialize filament lists
    filament_arcs = []
    filament_segments = []
    
    # First, add direct 2-3 connections as single-arc filaments
    for i, (pt1, pt2, segs) in enumerate(arcs):
        type1, type2 = point_types[pt1], point_types[pt2]
        if (type1 == 2 and type2 == 3) or (type1 == 3 and type2 == 2):
            filament_arcs.append([i])
            filament_segments.append(list(segs))
    
    # Then, for each type-4 component, find 2-3 filaments through type-4 nodes
    for component_4 in type4_components:
        # Find all 2s and 3s reachable from this component
        reachable_points = {2: [], 3: []}  # type -> [(point_id, path_arcs, entry_point)]
        
        # BFS from each node in the component to find reachable endpoints
        for start_node in component_4:
            visited = set()
            queue = deque([(start_node, [])])  # (node, path_of_arcs)
            
            while queue:
                curr_node, path = queue.popleft()
                if curr_node in visited:
                    continue
                visited.add(curr_node)
                
                for neighbor in graph[curr_node]:
                    if neighbor in visited:
                        continue
                    
                    # Get arc index
                    arc_key = (min(curr_node, neighbor), max(curr_node, neighbor))
                    arc_idx = arc_lookup[arc_key]
                    new_path = path + [arc_idx]
                    
                    neighbor_type = point_types[neighbor]
                    
                    if neighbor_type in [2, 3]:
                        # Found endpoint - record the path with entry point to component
                        reachable_points[neighbor_type].append((neighbor, new_path, start_node))
                    elif neighbor_type == 4 and neighbor in component_4 and len(path) == 0:
                        # Continue within the component only from the starting node
                        queue.append((neighbor, new_path))
        
        # Remove duplicates - keep shortest path to each endpoint
        for ptype in [2, 3]:
            unique_endpoints = {}
            for point_id, path, entry_point in reachable_points[ptype]:
                if point_id not in unique_endpoints or len(path) < len(unique_endpoints[point_id][1]):
                    unique_endpoints[point_id] = (point_id, path, entry_point)
            reachable_points[ptype] = list(unique_endpoints.values())
        
        # Create filaments between all unique 2-3 pairs
        for pt2, path2, entry2 in reachable_points[2]:
            for pt3, path3, entry3 in reachable_points[3]:
                
                # Find shortest path within component between entry points
                if entry2 == entry3:
                    # Same entry point - direct connection
                    bridge_path = []
                else:
                    # Find path within component
                    bridge_path = find_path_within_subset(entry2, entry3, 
                                                        component_4, graph, arc_lookup)
                    if bridge_path is None:
                        continue
                
                # Construct filament: reverse_path2 + bridge + path3
                filament_arc_list = []
                
                # Add path2 in reverse (from type-2 to component)
                for arc_idx in reversed(path2):
                    filament_arc_list.append(arc_idx)
                
                # Add bridge path within component
                filament_arc_list.extend(bridge_path)
                
                # Add path3 (from component to type-3)
                filament_arc_list.extend(path3)
                
                filament_arcs.append(filament_arc_list)
                
                # Construct segment sequence with proper direction handling
                seg_sequence = []
                
                # Build the complete node path to determine traversal directions
                node_path = [pt2]  # Start with type-2 point
                
                # Add nodes from path2 (in reverse since we're going from type-2 to component)
                current_node = pt2
                for arc_idx in reversed(path2):
                    pt1, pt2_arc, _ = arcs[arc_idx]
                    # Find which node is the next one in our path
                    next_node = pt2_arc if current_node == pt1 else pt1
                    node_path.append(next_node)
                    current_node = next_node
                
                # Add nodes from bridge path
                for arc_idx in bridge_path:
                    pt1, pt2_arc, _ = arcs[arc_idx]
                    next_node = pt2_arc if current_node == pt1 else pt1
                    node_path.append(next_node)
                    current_node = next_node
                
                # Add nodes from path3
                for arc_idx in path3:
                    pt1, pt2_arc, _ = arcs[arc_idx]
                    next_node = pt2_arc if current_node == pt1 else pt1
                    node_path.append(next_node)
                    current_node = next_node
                
                # Now build segments based on actual traversal direction
                all_arc_indices = list(reversed(path2)) + bridge_path + path3
                
                for i, arc_idx in enumerate(all_arc_indices):
                    pt1, pt2_arc, arc_segs = arcs[arc_idx]
                    
                    # Determine traversal direction
                    from_node = node_path[i]
                    to_node = node_path[i + 1]
                    
                    # Check if we're traversing the arc in its stored direction
                    if (pt1 == from_node and pt2_arc == to_node):
                        # Forward direction - use segments as stored
                        seg_sequence.extend(arc_segs)
                    elif (pt2_arc == from_node and pt1 == to_node):
                        # Reverse direction - reverse the segments
                        seg_sequence.extend(reversed(arc_segs))
                    else:
                        # This shouldn't happen if the path is correct
                        raise ValueError(f"Arc {arc_idx} doesn't connect nodes {from_node} and {to_node}")
                
                filament_segments.append(seg_sequence)
    
    return filament_arcs, filament_segments

def find_path_within_subset(start, end, subset, graph, arc_lookup):
    """Find shortest path between two nodes within a subset using BFS"""
    if start == end:
        return []
    
    visited = set()
    queue = deque([(start, [])])
    
    while queue:
        curr, path = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)
        
        if curr == end:
            return path
        
        # Only explore neighbors that are in the subset and connected in original graph
        for next_node in graph[curr]:
            if next_node in subset and next_node not in visited:
                arc_key = (min(curr, next_node), max(curr, next_node))
                if arc_key in arc_lookup:
                    new_path = path + [arc_lookup[arc_key]]
                    queue.append((next_node, new_path))
    
    return None  # No path found

filament_arcs, filament_segments = construct_filaments(arc_catalog, crits)

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if args.verbose:
    print('Filaments constructed..... ', formatted_time, flush = True)

################## Reformat filaments as paths ##################

def is_same_point(p1, p2, tolerance=1e-4):
    return np.allclose(p1, p2, atol=tolerance, rtol=0)
def segs_connected(idx1, idx2):
    if is_same_point(segs_Upos[idx1], segs_Upos[idx2]):
        return 'UU'
    if is_same_point(segs_Upos[idx1], segs_Vpos[idx2]):
        return 'UV'
    if is_same_point(segs_Vpos[idx1], segs_Upos[idx2]):
        return 'VU'
    if is_same_point(segs_Vpos[idx1], segs_Vpos[idx2]):
        return 'VV'
    return False
def fil_points(fil):
    points = []
    if len(fil) == 1:
        if segs_U_crit_type[fil[0]] == 2 and segs_V_crit_type[fil[0]] == 3:
            points.append([2, segs_Upos[fil[0]]])
            points.append([3, segs_Vpos[fil[0]]])
        elif segs_V_crit_type[fil[0]] == 2 and segs_U_crit_type[fil[0]] == 3:
            points.append([2, segs_Vpos[fil[0]]])
            points.append([3, segs_Upos[fil[0]]])
        else:
            print(segs_V_crit_type[fil[0]], segs_U_crit_type[fil[0]])
            raise ValueError('Filament not connecting 2-saddle and 3-maximum!')
        return points
    for i in range(len(fil) - 1):
        if not segs_connected(fil[i], fil[i + 1]):
            raise ValueError('Segments not connected in filament!')
            break
        conn = segs_connected(fil[i], fil[i + 1])
        if conn[0] == 'U':
            points.append([segs_V_crit_type[fil[i]], segs_Vpos[fil[i]]])
        else:
            points.append([segs_U_crit_type[fil[i]], segs_Upos[fil[i]]])

    if conn[1] == 'U':
        points.append([segs_U_crit_type[fil[-1]], segs_Upos[fil[-1]]])
        points.append([segs_V_crit_type[fil[-1]], segs_Vpos[fil[-1]]])
    else:
        points.append([segs_U_crit_type[fil[-1]], segs_Upos[fil[-1]]])
        points.append([segs_V_crit_type[fil[-1]], segs_Vpos[fil[-1]]])
    return points

filament_path = []
for fil in filament_segments:
    filament_path.append(fil_points(fil))

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if args.verbose:
    print('Filament paths created..... ', formatted_time, flush = True)

################## Save filament results ##################

with open(args.outname + "_filament_arcs.pkl", "wb") as f:
    pickle.dump(filament_arcs, f)
    
with open(args.outname + "_filament_segments.pkl", "wb") as f:
    pickle.dump(filament_segments, f)
    
with open(args.outname + "_filament_path.pkl", "wb") as f:
    pickle.dump(filament_path, f)

formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if args.verbose:
    print('Constuction done..... ', formatted_time, flush = True)


########################################################
################## Network statistics ##################
########################################################

################## Total length and filament length distribution ##################

def len_fil(fil):
    length = 0
    for i in range(len(fil) - 1):
        length += np.sqrt(np.sum((fil[i][1] - fil[i + 1][1]) ** 2))
    return length

fil_length = np.array([len_fil(fil) for fil in filament_path])

# sum all segments, not all filaments, do not include repetitive parts.
total_length = np.sum(segs_length[[s for arc in arc_catalog for s in arc[2]]])

################## Filament shape: curve ratio ##################

fil_straight_dist = np.array([np.sqrt(np.sum((fil[0][1] - fil[-1][1]) ** 2)) for fil in filament_path])
curve_ratio = fil_length / fil_straight_dist

################## Node connectivity ##################

nfil_per_node = np.unique([fil[-1][1] for fil in filament_path], axis = 0, return_counts = True)[1]

# Cell count and Gini coefficient for filament length and 2-saddle points (Emilie)


formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if args.verbose:
    print('Network statistics calculated..... ', formatted_time, flush = True)


if args.galfile is None:
    if args.verbose:
        print('No galaxy catalog supplied, saving network statistics only..... ')
    np.savez(args.outname + '_stats.npz',
             l_tot = total_length,
             l_arr = fil_length,
             curve_ratio = curve_ratio,
             nfil_per_node = nfil_per_node)

#########################################################################
################## Assignment of galaxies to filaments ##################
#########################################################################

else:
        
    ################## Load galaxies ##################
    
    dtype = [('gal_x', 'f8'), ('gal_y', 'f8'), ('gal_z', 'f8'), ('ms', 'f8')]
    gals = np.loadtxt(args.galfile, comments='#', usecols=(3, 4, 5, 8), dtype=dtype)
    gals['gal_x'] /= 1000
    gals['gal_y'] /= 1000
    gals['gal_z'] /= 1000
    # Axis flipping between galaxy catalog and cosmic web catalog, check for every case.
    gal_x = np.copy(gals['gal_x'])
    gal_y = np.copy(gals['gal_y'])
    gal_z = np.copy(gals['gal_z'])
    gals['gal_x'] = gal_z
    gals['gal_y'] = gal_x
    gals['gal_z'] = gal_y
    
    
    #dtype = [('ms', 'f8'), ('gal_x', 'f8'), ('gal_y', 'f8'), ('gal_z', 'f8'),]
    #gals = np.loadtxt('../SDSS/SDSS_z_44-476mpc_dr17.txt', comments='#', usecols=(9, 12, 13, 14), dtype=dtype)
    #gal_y = np.copy(gals['gal_y'])
    #gal_z = np.copy(gals['gal_z'])
    #gals['gal_y'] = gal_z
    #gals['gal_z'] = gal_y
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.verbose:
        print('Galaxies loaded..... ', formatted_time, flush = True)
    
    ################## Assign galaxies to segments ##################
    
    import time
    from numba import jit, prange
    from scipy.spatial import cKDTree
    import warnings
    
    @jit(nopython=True, parallel=True, fastmath=True)
    def point_to_segment_distance_numba(points, segment_starts, segment_ends):
        """
        Ultra-fast numba-compiled distance calculation with parallel processing.
        """
        n_points = points.shape[0]
        n_segments = segment_starts.shape[0]
        distances = np.empty((n_points, n_segments), dtype=np.float32)
        closest_points = np.empty((n_points, n_segments, 3), dtype=np.float32)
        
        for i in prange(n_points):
            for j in range(n_segments):
                # Vector from segment start to end
                dx = segment_ends[j, 0] - segment_starts[j, 0]
                dy = segment_ends[j, 1] - segment_starts[j, 1]
                dz = segment_ends[j, 2] - segment_starts[j, 2]
                
                # Vector from segment start to point
                px = points[i, 0] - segment_starts[j, 0]
                py = points[i, 1] - segment_starts[j, 1]
                pz = points[i, 2] - segment_starts[j, 2]
                
                # Length squared of segment
                line_len_sq = dx*dx + dy*dy + dz*dz
                
                if line_len_sq < 1e-12:
                    # Degenerate segment - distance to start point
                    distances[i, j] = np.sqrt(px*px + py*py + pz*pz)
                    closest_points[i, j, 0] = segment_starts[j, 0]
                    closest_points[i, j, 1] = segment_starts[j, 1]
                    closest_points[i, j, 2] = segment_starts[j, 2]
                else:
                    # Parameter t
                    dot_product = px*dx + py*dy + pz*dz
                    t = dot_product / line_len_sq
                    
                    # Clamp t to [0, 1]
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    
                    # Closest point on segment
                    closest_x = segment_starts[j, 0] + t * dx
                    closest_y = segment_starts[j, 1] + t * dy
                    closest_z = segment_starts[j, 2] + t * dz
                    
                    closest_points[i, j, 0] = closest_x
                    closest_points[i, j, 1] = closest_y
                    closest_points[i, j, 2] = closest_z
                    
                    # Distance
                    diff_x = points[i, 0] - closest_x
                    diff_y = points[i, 1] - closest_y
                    diff_z = points[i, 2] - closest_z
                    distances[i, j] = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        
        return distances, closest_points
    
    @jit(nopython=True)
    def find_min_distances_numba(distances):
        """Fast minimum finding with numba."""
        n_points = distances.shape[0]
        min_distances = np.empty(n_points, dtype=np.float32)
        min_indices = np.empty(n_points, dtype=np.int32)
        
        for i in range(n_points):
            min_dist = np.inf
            min_idx = 0
            for j in range(distances.shape[1]):
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    min_idx = j
            min_distances[i] = min_dist
            min_indices[i] = min_idx
        
        return min_distances, min_indices
    
    def build_spatial_index_segments(segments, grid_resolution=50):
        """
        Build a spatial grid index to quickly filter relevant segments.
        """
        # Extract U and V coordinates (start and end points)
        starts = np.column_stack([segments['U0'], segments['U1'], segments['U2']])
        ends = np.column_stack([segments['V0'], segments['V1'], segments['V2']])
        
        # Calculate bounding box
        all_points = np.vstack([starts, ends])
        bbox_min = np.min(all_points, axis=0)
        bbox_max = np.max(all_points, axis=0)
        
        # Create grid
        grid_size = (bbox_max - bbox_min) / grid_resolution
        
        # Assign segments to grid cells
        grid_dict = {}
        
        for seg_idx in range(len(segments)):
            start_point = starts[seg_idx]
            end_point = ends[seg_idx]
            
            # Find grid cells this segment touches
            seg_min = np.minimum(start_point, end_point)
            seg_max = np.maximum(start_point, end_point)
            
            grid_min = np.floor((seg_min - bbox_min) / grid_size).astype(int)
            grid_max = np.floor((seg_max - bbox_min) / grid_size).astype(int)
            
            # Clamp to valid range
            grid_min = np.maximum(grid_min, 0)
            grid_max = np.minimum(grid_max, grid_resolution - 1)
            
            # Add to all touched grid cells
            for x in range(grid_min[0], grid_max[0] + 1):
                for y in range(grid_min[1], grid_max[1] + 1):
                    for z in range(grid_min[2], grid_max[2] + 1):
                        cell_key = (x, y, z)
                        if cell_key not in grid_dict:
                            grid_dict[cell_key] = []
                        grid_dict[cell_key].append(seg_idx)
        
        return {
            'grid_dict': grid_dict,
            'bbox_min': bbox_min,
            'grid_size': grid_size,
            'resolution': grid_resolution
        }
    
    def get_relevant_segments(points, spatial_index, search_radius_cells=1):
        """
        Get segments that could potentially be closest to given points.
        """
        grid_dict = spatial_index['grid_dict']
        bbox_min = spatial_index['bbox_min']
        grid_size = spatial_index['grid_size']
        resolution = spatial_index['resolution']
        
        all_relevant_segments = set()
        
        for point in points:
            # Find grid cell for this point
            grid_pos = np.floor((point - bbox_min) / grid_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, resolution - 1)
            
            # Search in neighboring cells
            for dx in range(-search_radius_cells, search_radius_cells + 1):
                for dy in range(-search_radius_cells, search_radius_cells + 1):
                    for dz in range(-search_radius_cells, search_radius_cells + 1):
                        cell_key = (
                            grid_pos[0] + dx,
                            grid_pos[1] + dy,
                            grid_pos[2] + dz
                        )
                        
                        # Check bounds
                        if (0 <= cell_key[0] < resolution and 
                            0 <= cell_key[1] < resolution and 
                            0 <= cell_key[2] < resolution):
                            
                            if cell_key in grid_dict:
                                all_relevant_segments.update(grid_dict[cell_key])
        
        return list(all_relevant_segments)
    
    def assign_points_to_segments_ultra_fast(points, segments, use_spatial_index=True, 
                                            max_points_per_chunk=50000, max_segments_per_chunk=50000,
                                            exact_precision=False, spatial_search_radius=2):
        """
        Ultra-fast assignment of points to closest segments using numba compilation and spatial indexing.
        
        Parameters:
        - points: Array of shape (N, 3) with point coordinates
        - segments: Structured array with fields U0,U1,U2,V0,V1,V2 for segment endpoints
        - use_spatial_index: If True, uses spatial indexing for faster queries
        - exact_precision: If True, uses float64 and disables spatial indexing for exact results
        - spatial_search_radius: How many grid cells to search around each point (higher = more accurate)
        
        Returns:
        - segment_assignments: Array of segment indices (one per point)
        - min_distances: Array of distances to closest segment (one per point)
        - connection_points: Array of closest points on segments (N x 3)
        """
        if args.verbose:
            print(f"Dataset size: {len(points)} points, {len(segments)} segments")
        
        # Choose precision based on exact_precision flag
        dtype = np.float64 if exact_precision else np.float32
        points = np.asarray(points, dtype=dtype)
        
        # Disable spatial indexing if exact precision is requested
        if exact_precision:
            use_spatial_index = False
            if args.verbose:
                print("Exact precision mode: using float64 and disabling spatial indexing")
        
        # Build spatial index if requested
        spatial_index = None
        if use_spatial_index:
            if args.verbose:
                print("Building spatial index...")
            start_time = time.time()
            spatial_index = build_spatial_index_segments(segments)
            if args.verbose:
                print(f"Spatial index built in {time.time() - start_time:.2f} seconds")
        
        # Extract segment endpoints
        if args.verbose:
            print("Extracting segment endpoints...")
        segment_starts = np.column_stack([segments['U0'], segments['U1'], segments['U2']]).astype(dtype)
        segment_ends = np.column_stack([segments['V0'], segments['V1'], segments['V2']]).astype(dtype)
        
        total_segments = len(segments)
        num_points = len(points)
        
        if args.verbose:
            print(f"Processing {num_points} points against {total_segments} segments")
        
        # Initialize results
        segment_assignments = np.full(num_points, -1, dtype=np.int32)
        min_distances = np.full(num_points, np.inf, dtype=dtype)
        connection_points = np.zeros((num_points, 3), dtype=dtype)
        
        start_time = time.time()
        
        # Process in chunks
        for point_start in range(0, num_points, max_points_per_chunk):
            point_end = min(point_start + max_points_per_chunk, num_points)
            point_chunk = points[point_start:point_end]
            chunk_size = point_end - point_start
            
            if args.verbose:
                print(f"Processing points {point_start:6d}-{point_end:6d}")
            
            # Get relevant segments if using spatial index
            if use_spatial_index:
                relevant_seg_indices = get_relevant_segments(point_chunk, spatial_index, 
                                                           search_radius_cells=spatial_search_radius)
                if len(relevant_seg_indices) == 0:
                    if args.verbose:
                        print("  No relevant segments found, using all segments")
                    relevant_seg_indices = list(range(total_segments))
            else:
                relevant_seg_indices = list(range(total_segments))
            
            if args.verbose:
                print(f"  Checking against {len(relevant_seg_indices)} relevant segments")
            
            # Process relevant segments in chunks
            chunk_min_distances = np.full(chunk_size, np.inf, dtype=dtype)
            chunk_assignments = np.full(chunk_size, -1, dtype=np.int32)
            chunk_connections = np.zeros((chunk_size, 3), dtype=dtype)
            
            for seg_start in range(0, len(relevant_seg_indices), max_segments_per_chunk):
                seg_end = min(seg_start + max_segments_per_chunk, len(relevant_seg_indices))
                
                # Get segment indices for this chunk
                seg_chunk_indices = relevant_seg_indices[seg_start:seg_end]
                seg_starts_chunk = segment_starts[seg_chunk_indices]
                seg_ends_chunk = segment_ends[seg_chunk_indices]
                
                # Calculate distances using numba
                distances, closest_points = point_to_segment_distance_numba(
                    point_chunk, seg_starts_chunk, seg_ends_chunk
                )
                
                # Find minimums using numba
                min_dist_in_chunk, min_seg_in_chunk = find_min_distances_numba(distances)
                
                # Update global minimums
                update_mask = min_dist_in_chunk < chunk_min_distances
                chunk_min_distances[update_mask] = min_dist_in_chunk[update_mask]
                
                # Update assignments and connections
                for i in np.where(update_mask)[0]:
                    global_seg_idx = seg_chunk_indices[min_seg_in_chunk[i]]
                    chunk_assignments[i] = global_seg_idx
                    chunk_connections[i] = closest_points[i, min_seg_in_chunk[i]]
            
            # Store results
            segment_assignments[point_start:point_end] = chunk_assignments
            min_distances[point_start:point_end] = chunk_min_distances
            connection_points[point_start:point_end] = chunk_connections
        
        elapsed = time.time() - start_time
        
        # Results summary
        if args.verbose:
            print(f"\nProcessing complete!")
            print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            
        unassigned = np.sum(segment_assignments == -1)
        if args.verbose:
            if unassigned > 0:
                print(f"Warning: {unassigned} points remain unassigned")
            
            print(f"Results Summary:")
            print(f"  Average distance: {np.mean(min_distances):.4f}")
            print(f"  Max distance: {np.max(min_distances):.4f}")
            print(f"  Min distance: {np.min(min_distances):.4f}")
            
        unique_assignments, counts = np.unique(segment_assignments[segment_assignments != -1], return_counts=True)
        if args.verbose:
            if len(unique_assignments) > 0:
                print(f"  Points assigned to {len(unique_assignments)} different segments")
                print(f"  Most popular segment has {np.max(counts)} points")
                print(f"  Least popular segment has {np.min(counts)} points")
            
        return segment_assignments, min_distances, connection_points
    
    def run_ultra_fast_segment_assignment(points, segments, exact_results=False, **kwargs):
        """
        Convenience function for segment assignment with optimal default settings.
        
        Parameters:
        - points: Array of shape (N, 3) with point coordinates
        - segments: Structured array with U0,U1,U2,V0,V1,V2 fields
        - exact_results: If True, guarantees exact results (slower but precise)
        
        Returns:
        - segment_indices: Index of closest segment for each point
        - distances: Distance to closest segment for each point  
        - connection_points: Closest point on segment for each point
        """
        if exact_results:
            # Settings for exact reproduction of results
            default_kwargs = {
                'use_spatial_index': False,  # Disable spatial indexing for exact results
                'exact_precision': True,     # Use float64 precision
                'max_points_per_chunk': 5000,
                'max_segments_per_chunk': 10000
            }
            if args.verbose:
                print("Exact results mode: This will be slower but give precise results")
        else:
            # Optimal settings for speed (may have tiny numerical differences)
            default_kwargs = {
                'use_spatial_index': True,
                'exact_precision': False,
                'spatial_search_radius': 2,  # Increased for better accuracy
                'max_points_per_chunk': 10000,
                'max_segments_per_chunk': 20000
            }
        
        default_kwargs.update(kwargs)
        
        return assign_points_to_segments_ultra_fast(points, segments, **default_kwargs)
    
    segment_indices, distances, connections = run_ultra_fast_segment_assignment(
            np.vstack([gals['gal_x'], gals['gal_y'], gals['gal_z']]).T, segs, exact_results=True)
        
    if args.verbose:
        print(f"\nExample results:")
        print(f"First 5 points assigned to segments: {segment_indices[:5]}")
        print(f"First 5 distances: {distances[:5]}")
        print(f"First connection point: {connections[0]}")
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.verbose:
        print('Galaxies assigned to segments..... ', formatted_time, flush = True)
    
    ################## Reorganize assignment around filaments ##################
    
    def assign_galaxies_to_filaments(segment_indices, distances, connections, points, filament_segments):
        
        if args.verbose:
            print(f"Processing {len(points)} galaxies across {len(filament_segments)} filaments...")
        
        # Build mapping from segment to filament(s)
        segment_to_filaments = defaultdict(list)
        for filament_idx, segments in enumerate(filament_segments):
            for segment_idx in segments:
                segment_to_filaments[segment_idx].append(filament_idx)
        
        if args.verbose:
            print(f"Segment-to-filament mapping built: {len(segment_to_filaments)} unique segments")
        
        # Initialize results for each filament
        filament_galaxies = []
        for filament_idx in range(len(filament_segments)):
            filament_galaxies.append({
                'galaxy_indices': [],
                'galaxy_coords': [],
                'distances': [],
                'connections': [],
                'segment_indices': []
            })
        
        # Assign each galaxy to its relevant filament(s)
        for galaxy_idx, closest_segment in enumerate(segment_indices):
            if closest_segment == -1:
                continue  # Skip unassigned galaxies
            
            # Find all filaments that contain this segment
            filament_list = segment_to_filaments.get(closest_segment, [])
            
            # Add this galaxy to all relevant filaments
            for filament_idx in filament_list:
                filament_galaxies[filament_idx]['galaxy_indices'].append(galaxy_idx)
                filament_galaxies[filament_idx]['galaxy_coords'].append(points[galaxy_idx])
                filament_galaxies[filament_idx]['distances'].append(distances[galaxy_idx])
                filament_galaxies[filament_idx]['connections'].append(connections[galaxy_idx])
                filament_galaxies[filament_idx]['segment_indices'].append(closest_segment)
        
        # Convert lists to numpy arrays for better performance
        for filament_idx in range(len(filament_segments)):
            filament_data = filament_galaxies[filament_idx]
            filament_data['galaxy_indices'] = np.array(filament_data['galaxy_indices'], dtype=np.int32)
            filament_data['galaxy_coords'] = np.array(filament_data['galaxy_coords'], dtype=np.float64)
            filament_data['distances'] = np.array(filament_data['distances'], dtype=np.float64)
            filament_data['connections'] = np.array(filament_data['connections'], dtype=np.float64)
            filament_data['segment_indices'] = np.array(filament_data['segment_indices'], dtype=np.int32)
        
        # Print summary statistics
        if args.verbose:
            print(f"\nFilament assignment summary:")
        total_assignments = 0
        non_empty_filaments = 0
        max_galaxies = 0
        min_galaxies = float('inf')
        
        for filament_idx, filament_data in enumerate(filament_galaxies):
            n_galaxies = len(filament_data['galaxy_indices'])
            total_assignments += n_galaxies
            
            if n_galaxies > 0:
                non_empty_filaments += 1
                max_galaxies = max(max_galaxies, n_galaxies)
                min_galaxies = min(min_galaxies, n_galaxies)
        
        if non_empty_filaments > 0:
            avg_galaxies = total_assignments / non_empty_filaments
            if args.verbose:
                print(f"  {non_empty_filaments}/{len(filament_segments)} filaments have galaxies")
                print(f"  Average galaxies per non-empty filament: {avg_galaxies:.1f}")
                print(f"  Min galaxies in a filament: {min_galaxies}")
                print(f"  Max galaxies in a filament: {max_galaxies}")
                print(f"  Total galaxy-filament associations: {total_assignments}")
                
            # Check for shared segments
            segments_in_multiple = sum(1 for seg_filaments in segment_to_filaments.values() 
                                     if len(seg_filaments) > 1)
            if args.verbose:
                print(f"  {segments_in_multiple} segments belong to multiple filaments")
        else:
            if args.verbose:
                print("  No galaxies assigned to any filaments!")
        
        return filament_galaxies
    
    def get_filament_statistics(filament_galaxies, filament_idx):
        """
        Get detailed statistics for a specific filament.
        
        Parameters:
        - filament_galaxies: Output from assign_galaxies_to_filaments
        - filament_idx: Index of the filament to analyze
        
        Returns:
        - Dictionary with detailed statistics
        """
        if filament_idx >= len(filament_galaxies):
            raise ValueError(f"Filament index {filament_idx} out of range")
        
        filament_data = filament_galaxies[filament_idx]
        n_galaxies = len(filament_data['galaxy_indices'])
        
        if n_galaxies == 0:
            return {
                'n_galaxies': 0,
                'mean_distance': None,
                'std_distance': None,
                'min_distance': None,
                'max_distance': None,
                'unique_segments': None,
                'galaxy_density': None
            }
        
        distances = filament_data['distances']
        unique_segments = len(np.unique(filament_data['segment_indices']))
        
        # Calculate bounding box volume for density
        coords = filament_data['galaxy_coords']
        bbox_volume = np.prod(np.max(coords, axis=0) - np.min(coords, axis=0))
        galaxy_density = n_galaxies / bbox_volume if bbox_volume > 0 else None
        
        return {
            'n_galaxies': n_galaxies,
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'unique_segments': unique_segments,
            'galaxy_density': galaxy_density,
            'bbox_volume': bbox_volume
        }
    
    def find_galaxies_near_filament(filament_galaxies, filament_idx, max_distance=None, 
                                   n_closest=None, return_indices_only=False):
        """
        Find galaxies associated with a filament based on distance criteria.
        
        Parameters:
        - filament_galaxies: Output from assign_galaxies_to_filaments
        - filament_idx: Index of the filament
        - max_distance: Only return galaxies within this distance (optional)
        - n_closest: Only return the N closest galaxies (optional)
        - return_indices_only: If True, only return galaxy indices
        
        Returns:
        - Filtered subset of filament data or just indices
        """
        if filament_idx >= len(filament_galaxies):
            raise ValueError(f"Filament index {filament_idx} out of range")
        
        filament_data = filament_galaxies[filament_idx]
        distances = filament_data['distances']
        
        if len(distances) == 0:
            return [] if return_indices_only else {
                'galaxy_indices': np.array([], dtype=np.int32),
                'galaxy_coords': np.array([]).reshape(0, 3),
                'distances': np.array([]),
                'connections': np.array([]).reshape(0, 3),
                'segment_indices': np.array([], dtype=np.int32)
            }
        
        # Apply distance filter
        mask = np.ones(len(distances), dtype=bool)
        if max_distance is not None:
            mask &= (distances <= max_distance)
        
        # Apply n_closest filter
        if n_closest is not None and n_closest < len(distances):
            if max_distance is not None:
                # Apply distance filter first, then take closest
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > n_closest:
                    closest_indices = valid_indices[np.argsort(distances[valid_indices])[:n_closest]]
                    mask = np.zeros(len(distances), dtype=bool)
                    mask[closest_indices] = True
            else:
                # Just take n closest overall
                closest_indices = np.argsort(distances)[:n_closest]
                mask = np.zeros(len(distances), dtype=bool)
                mask[closest_indices] = True
        
        if return_indices_only:
            return filament_data['galaxy_indices'][mask]
        
        return {
            'galaxy_indices': filament_data['galaxy_indices'][mask],
            'galaxy_coords': filament_data['galaxy_coords'][mask],
            'distances': filament_data['distances'][mask],
            'connections': filament_data['connections'][mask],
            'segment_indices': filament_data['segment_indices'][mask]
        }
    
    filament_galaxies_no_truncation = assign_galaxies_to_filaments(
            segment_indices, distances, connections,
            np.vstack([gals['gal_x'], gals['gal_y'], gals['gal_z']]).T, filament_segments)
    
    filament_galaxies = []
    for fil in filament_galaxies_no_truncation:
        fil_ = {}
        mask = fil['distances'] <= 2
        for key in fil.keys():
            fil_[key] = fil[key][mask]
        filament_galaxies.append(fil_)
    
    with open(args.outname + "_filament_galaxies_no_truncation.pkl", "wb") as f:
        pickle.dump(filament_galaxies_no_truncation, f)
        
    with open(args.outname + "_filament_galaxies.pkl", "wb") as f:
        pickle.dump(filament_galaxies, f)
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.verbose:
        print('Filament galaxy assignment done..... ', formatted_time, flush = True)
    
    ############################################################################
    ################## Galaxy-filament association statistics ##################
    ############################################################################
    
    ################## Galaxy-filament distance distribution ##################
    
    distances_all = np.hstack([fil['distances'] for fil in filament_galaxies])
    distances_all_no_truncation = np.hstack([fil['distances'] for fil in filament_galaxies_no_truncation])
    
    ################## Filament richness ##################
    
    richness = np.array([len(fil['galaxy_indices']) for fil in filament_galaxies])
    
    ################## Filament stellar mass richness ##################
    
    mass_richness = np.array([np.sum(gals['ms'][fil['galaxy_indices']]) for fil in filament_galaxies])
    
    ################## Distribution of connection point location on filament ##################
    
    # distance from 2-saddle point
    dist_2 = [np.array([np.sum(np.sqrt(np.sum(np.square(np.diff(
                    np.vstack([[pt[1] for pt in
                                filament_path[i][: np.where(filament_segments[i] == filament_galaxies[i]['segment_indices'][j])[0][0] + 1]],
                               filament_galaxies[i]['connections'][j]]), axis = 0)), axis = 1)))
                        for j in range(len(filament_galaxies[i]['segment_indices']))]) for i in range(len(filament_galaxies))]
    dist_2 = np.concatenate(dist_2)
    # distance from maximum
    dist_3 = [np.array([np.sum(np.sqrt(np.sum(np.square(np.diff(
                    np.vstack([filament_galaxies[i]['connections'][j],
                               [pt[1] for pt in
                                filament_path[i][np.where(filament_segments[i] == filament_galaxies[i]['segment_indices'][j])[0][0] + 1 :]]]),
                                axis = 0)), axis = 1)))
                        for j in range(len(filament_galaxies[i]['segment_indices']))]) for i in range(len(filament_galaxies))]
    dist_3 = np.concatenate(dist_3)
    
    ################## Richness-length correlation and ratio ##################
    
    corr_len_richness = np.corrcoef(fil_length, richness)[0, 1]
    corr_len_mass_richness = np.corrcoef(fil_length, mass_richness)[0, 1]
    
    richness_per_length = richness / fil_length
    mass_richness_per_length = mass_richness / fil_length
    
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.verbose:
        print('Galaxy-filament association statistics done..... ', formatted_time, flush = True)
    
    ##################################################
    ################## Save to file ##################
    ##################################################
    
    np.savez(args.outname + '_stats.npz',
             l_tot = total_length,
             l_arr = fil_length,
             curve_ratio = curve_ratio,
             nfil_per_node = nfil_per_node,
             distances_all_no_truncation = distances_all_no_truncation,
             distances_all = distances_all,
             richness = richness,
             mass_richness = mass_richness,
             dist_2 = dist_2,
             dist_3 = dist_3,
             corr_len_richness = corr_len_richness,
             corr_len_mass_richness = corr_len_mass_richness,
             richness_per_length = richness_per_length,
             mass_richness_per_length = mass_richness_per_length)
    
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.verbose:
        print('Results saved..... ', formatted_time, flush = True)
