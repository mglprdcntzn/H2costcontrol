import numpy as np
import random
import math
import matplotlib.pyplot as plt
#############################################################
def load_circuit(N, Mean, dev,classes):
    load = np.array([random.gauss(Mean, dev) for _ in range(N)])
    load = abs(load)  #eliminate negativess
    loadmix = np.random.dirichlet(np.ones(classes), N)
    
    return load, loadmix
#############################################################
def DG_circuit(N, Mean, dev):
    
    dg = np.array([random.gauss(Mean, dev) for _ in range(N)])
    dg = abs(dg)  #eliminate negatives
    return dg

#############################################################
def impendances_circuit(lines, N, NPV):
    ###############
    NB = lines.shape[0] #num of 
    lines_info = np.zeros((NB, 3),dtype='object') #an array to save the info
    ###############
    Rperkm = 0.2870120
    Xperkm = 0.5508298
    Rpermt = Rperkm / 1000
    Xpermt = Xperkm / 1000
    ###############
    epsilon = 1/100
    upper = 1 + epsilon
    lower = 1- epsilon
    ###############
    W = np.zeros((NB, NB), dtype=complex)  #admitances of each line
    D = np.zeros((NB, N), dtype=int)  #incidence matrix
    
    for ll in range(NB):
        origin  = int(lines[ll, 0]) 
        destiny = int(lines[ll, 1]) 
        Rll = Rpermt
        Xll = Xpermt
        #impedance
        Z = lines[ll, 2] * ( Rll* random.uniform(lower, upper) + 1j * Xll * random.uniform(lower, upper) )
        #admittance
        W[ll, ll] = 1 / Z
        #incidences
        D[ll, origin] = -1
        D[ll, destiny] = 1
        #lineinfo
        if origin<destiny:
            lines_info[ll,0] = origin
            lines_info[ll,1] = destiny
        else:
            lines_info[ll,1] = origin
            lines_info[ll,0] = destiny
        lines_info[ll,2] = Z
    
    #sort info table
    lines_info = lines_info[np.lexsort((lines_info[:, 1], lines_info[:, 0]))]
    #admitances matrix
    hatY = np.transpose(D) @ W @ D
    
    #partitionate adm matrx
    Y00 = hatY[0:NPV, 0:NPV]
    Y0  = hatY[NPV:N, 0:NPV]
    Y   = hatY[NPV:N, NPV:N]
    
    return Y, Y0, Y00, lines_info
#############################################################
def print_circuit(nodes, lines, plt_name,indexes_reg,indexes_sol,indexes_hid,indexes_load):
    #prepare figure
    fig, ax = plt.subplots(figsize=(10 * 2 / 2.54, 10 * 2 / 2.54))
    
    #ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    #nodes
    yoffset = 40
    ax.scatter(nodes[indexes_reg, 0], nodes[indexes_reg, 1]+yoffset, color='brown',zorder=3)
    ax.scatter(nodes[indexes_sol, 0], nodes[indexes_sol, 1]+yoffset, color='orange',zorder=3)
    ax.scatter(nodes[indexes_hid, 0], nodes[indexes_hid, 1]+yoffset, color='blue',zorder=3)
    ax.scatter(nodes[indexes_load, 0], nodes[indexes_load, 1]+yoffset, color='black',zorder=3)
    #nodes labels
    dx = 0.04
    dy = 0.04
    for idx in indexes_reg:
        x = nodes[idx, 0]
        y = nodes[idx, 1] + yoffset
        ax.text(x+dx,y+dy, str(idx), fontsize=9, ha='left', va='bottom',color='brown')
    for idx in indexes_sol:
        x = nodes[idx, 0]
        y = nodes[idx, 1] + yoffset
        ax.text(x+dx,y+dy, str(idx), fontsize=9, ha='left', va='bottom',color='orange')
    for idx in indexes_hid:
        x = nodes[idx, 0]
        y = nodes[idx, 1] + yoffset
        ax.text(x+dx,y+dy, str(idx), fontsize=9, ha='left', va='bottom',color='blue')
    for idx in indexes_load:
        x = nodes[idx, 0]
        y = nodes[idx, 1] + yoffset
        ax.text(x+dx,y+dy, str(idx), fontsize=9, ha='left', va='bottom',color='black')
    
    #lines
    rows, cols = lines.shape
    for ll in range(0, rows):
        xx = np.array([
          nodes[lines[ll, 0].astype(int), 0],
          nodes[lines[ll, 1].astype(int), 0]
        ])
        yy = np.array([
          nodes[lines[ll, 0].astype(int), 1]+yoffset,
          nodes[lines[ll, 1].astype(int), 1]+yoffset
        ])
        ax.plot(xx, yy, linestyle='-', color='gray',zorder=2)
    
    ax.set_aspect('equal', adjustable='box')
    #find corners coordinates
    left, right = ax.get_xlim()
    lower,upper = ax.get_ylim()
    #draw a light grid
    for x in np.arange(left+1000, right, 1000):
        ax.vlines(x, lower, upper, colors='lightgray', linestyles='--', linewidth=0.5, zorder=1)
    for y in np.arange(lower+1000, upper, 1000):
        ax.hlines(y, left, right, colors='lightgray', linestyles='--', linewidth=0.5, zorder=1)
        
    ax.vlines(left, lower, upper, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    ax.vlines(right, lower, upper, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    ax.hlines(lower, left, right, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    ax.hlines(upper, left, right, colors='gray', linestyles='-', linewidth=0.5, zorder=1)
    
    #draw scale reference at lower left corner
    scale0x = left #+ 20
    scale0y = lower #- 20
    ax.plot([scale0x,scale0x+1000], [scale0y,scale0y], linestyle='-', color='black')
    ax.plot([scale0x,scale0x], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+250,scale0x+250], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+500,scale0x+500], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+750,scale0x+750], [scale0y+50,scale0y-50], linestyle='-', color='black')
    ax.plot([scale0x+1000,scale0x+1000], [scale0y+50,scale0y-50], linestyle='-', color='black')
    
    ax.text(scale0x+000, scale0y-70, '0m'  , fontsize=8, ha='center', va='top',color='black') 
    # ax.text(scale0x+500, scale0y-70, '500m', fontsize=8, ha='center', va='top',color='black') 
    ax.text(scale0x+1000, scale0y-70, '1000m', fontsize=8, ha='center', va='top',color='black') 
    
    #draw dots with legend under scale
    legendx = scale0x
    legendy = scale0y - 500
    
    ax.plot(legendx, legendy, 'o', color='brown')
    ax.plot(legendx, legendy-260, 'o', color='orange')
    ax.plot(legendx, legendy-520, 'o', color='blue')
    ax.plot(legendx, legendy-780, 'o', color='black')
    
    ax.text(legendx+90, legendy, ': Regulation nodes', fontsize=12, ha='left', va='center',color='black') 
    ax.text(legendx+90, legendy-260, ': Photo-voltaic generation nodes', fontsize=12, ha='left', va='center',color='black') 
    ax.text(legendx+90, legendy-520, ': Hydrogen production nodes', fontsize=12, ha='left', va='center',color='black') 
    ax.text(legendx+90, legendy-780, ': Industrial load nodes', fontsize=12, ha='left', va='center',color='black') 
    
    ax.set_frame_on(False)
    plt.tight_layout()
    # Saving the plot to an image file
    fig.savefig(plt_name+'.eps', format='eps')
    plt.show()
    fig.clf()
    return

#############################################################
def create_tree_circuit(N, Dmean, Dmin):
    #prefill vectors
    nodes = np.zeros((N, 2))  #x,y
    lines = np.empty((0, 3))#norigin,ndestiny,distance

    breakprob  = 0.50

    maxangle   = 10*math.pi/180
    breakangle = 0
    std_dev    = (Dmean-Dmin)/2
    
    nor   = 0
    #run over the nodes
    for ii in range(N-1):
        ndes = ii + 1
        
        if nor>0:
            if random.random() < breakprob:
                nor        = random.randint(1, ndes-1)
                if breakangle<0:
                    breakangle = random.choice([0,1])*90*math.pi/180 # random.randint(0, 3)*90*math.pi/180
                elif breakangle>0:
                    breakangle = random.choice([-1,0])*90*math.pi/180 # random.randint(0, 3)*90*math.pi/180
                else:
                    breakangle = 90*math.pi/180#random.choice([-1,1])*90*math.pi/180 # random.randint(0, 3)*90*math.pi/180
            # else:
            #     breakangle = 0
            
        xor  = nodes[nor,0]
        yor  = nodes[nor,1]
        
        searching = True
        while searching:
            theta = random.random() * 2 * maxangle - maxangle + breakangle
            
            factor = -2*(breakangle/math.pi)**2 + 3/2
            dist  = random.gauss(Dmean*factor, std_dev)
            
            x = xor +  dist*np.cos(theta)
            y = yor +  dist*np.sin(theta)
            
            #check distances
            checking = True
            for jj in range(ndes):
                d = np.sqrt((x-nodes[jj,0])**2 + (y-nodes[jj,1])**2)
                if d<Dmin:
                    checking = False
                    break
            #check intersections
            checking = checking and not(intersected_lines(nor,[x,y],lines,nodes))
            
            if checking:
                searching = False
                break
            else:
                nor        = random.randint(1, ndes-1)
                breakangle = random.choice([-1,0,0,1])*90*math.pi/180 #random.randint(0, 3)*90*math.pi/180
                
                xor        = nodes[nor,0]
                yor        = nodes[nor,1]
                
        nodes[ndes,:] = [x,y]
        lines         = np.vstack([lines, [nor,ndes,dist]])
        nor           = ndes
        
    return nodes, lines

#############################################################
def add_loops_to_circuit(nodes, lines, Dmean, Dmin):
    N = nodes.shape[0] #number of nodes
    p = 0.3
    #find neighbours and number
    neighbours = np.zeros((N,N))
    for line in lines:
        nor  = int(line[0])
        ndes = int(line[1])
        
        neighbours[nor][ndes] = 1
        neighbours[ndes][nor] = 1
    neig_number = neighbours.sum(axis=1)
    #add loops from end nodes to a near one
    for ii in range(N):
        if neig_number[ii]==1:
            xii = nodes[ii,0]
            yii = nodes[ii,1]
            
            for jj in range(N):
                if neighbours[ii][jj]!=1 and ii!=jj:
                    xjj = nodes[jj,0]
                    yjj = nodes[jj,1]
                    
                    dd  = np.sqrt((xii-xjj)**2 + (yii-yjj)**2)
                    
                    if Dmin < dd < (1-p)*Dmin+p*Dmean:
                        if not(intersected_lines(ii,[xjj,yjj],lines,nodes)):
                            lines = np.vstack([lines, [ii,jj,dd]])
                            
                            neighbours[ii][jj] = 1
                            neighbours[jj][ii] = 1
                            neig_number        = neighbours.sum(axis=1)
    return lines
#############################################################
def intersected_lines(node,newxy,lines,nodes):
    #coordinates of new point
    xi = nodes[node, 0]
    yi = nodes[node, 1]
    xj = newxy[0]
    yj = newxy[1]
    #go through existing segments
    for ll in lines:
        #points of existing segments
        aa  = int(ll[0])
        bb  = int(ll[1])
        #coordinates of existing segments
        xa  = nodes[aa, 0]
        ya  = nodes[aa, 1]
        xb  = nodes[bb, 0]
        yb  = nodes[bb, 1]
        #solve parametric representation of both segments
        den = (xb-xa)*(yj-yi) - (xj-xi)*(yb-ya)
        if den != 0:
            t = ( (xi-xa)*(yb-ya) - (xb-xa)*(yi-ya) ) / den
            u = ( (xj-xi)*(ya-yi) - (xa-xi)*(yj-yi) ) / den
            
            if 0<t<1 and 0<u<1:
                #intersection!
                return True
    return False

#############################################################
def choose_nodes(nodes, lines, NPV, Nsol, Nhid):
    N = nodes.shape[0] #number of nodes
    #mean position of all nodes
    Ex, Ey = np.mean(nodes, axis=0)
    #define a list of free nodes
    indexes_free  = list(set(range(N)))
    #find neighbours number
    neighbours = np.zeros((N,N))
    for line in lines:
        nor  = int(line[0])
        ndes = int(line[1])
        
        neighbours[nor][ndes] = 1
        neighbours[ndes][nor] = 1
    neig_number = neighbours.sum(axis=1)
    #indexes of nodes with just one neigh
    indexes_lonely = list(set(np.where(neig_number == 1)[0]))
    indexes_free   = list(set(indexes_free) - set(indexes_lonely))
    #reorder indexes from the furthest away to the closest to center
    distances      = np.linalg.norm(nodes[indexes_lonely] - np.array([Ex, Ey]), axis=1)
    indexes_lonely = [i for _, i in sorted(zip(distances, indexes_lonely), reverse=True)]
    #randomly choose regulation nodes with onlye 1 neigh
    if len(indexes_lonely)>=NPV:
        # indexes_reg    = random.sample(indexes_lonely, NPV)
        indexes_reg    = indexes_lonely[:NPV]
    else:
        # indexes_reg    = list(set(indexes_lonely) | set(random.sample(indexes_free, NPV-len(indexes_lonely) )))
        indexes_reg    = indexes_lonely[:NPV-len(indexes_lonely)]
    
    indexes_free   = list(set(indexes_free) - set(indexes_reg))
    indexes_lonely = list(set(indexes_lonely) - set(indexes_reg))
    #choose random sun nodes
    if len(indexes_lonely)>=Nsol:
        indexes_sol    = random.sample(indexes_lonely, Nsol)
    else:
        indexes_sol    = list(set(indexes_lonely) | set(random.sample(indexes_free, Nsol-len(indexes_lonely) )))
    
    indexes_free   = list(set(indexes_free) - set(indexes_sol))
    indexes_lonely = list(set(indexes_lonely) - set(indexes_sol))
        
    #choose random hid nodes
    if len(indexes_lonely)>=Nhid:
        indexes_hid    = random.sample(indexes_lonely, Nhid)
    else:
        indexes_hid    = list(set(indexes_lonely) | set(random.sample(indexes_free, Nhid-len(indexes_lonely) )))
    
    indexes_free   = list(set(indexes_free) - set(indexes_hid))
    indexes_lonely = list(set(indexes_lonely) - set(indexes_hid))
    
    #set the rest of free nodes as loads
    indexes_lds    = list(indexes_free)
    
    return indexes_reg, indexes_hid, indexes_sol, indexes_lds

#############################################################
def reorder_nodes(nodes, lines, indexes_reg,indexes_sol,indexes_hid,indexes_lds):
    new_nodes = np.zeros(nodes.shape)
    new_lines = np.zeros(lines.shape)
    
    indexes_translate = np.zeros(nodes.shape[0])
    
    new_indexes_reg  = []
    new_indexes_sol  = []
    new_indexes_hid  = []
    new_indexes_lds = []
    
    new_ii = 0
    for ii in indexes_reg:
        new_nodes[new_ii,:] = nodes[ii,:]
        new_indexes_reg.append(new_ii)
        indexes_translate[ii] = new_ii
                
        new_ii = new_ii + 1
    
    for ii in indexes_hid:
        new_nodes[new_ii,:] = nodes[ii,:]
        new_indexes_hid.append(new_ii)
        indexes_translate[ii] = new_ii
                
        new_ii = new_ii + 1
    
    for ii in indexes_sol:
        new_nodes[new_ii,:] = nodes[ii,:]
        new_indexes_sol.append(new_ii)
        indexes_translate[ii] = new_ii
                
        new_ii = new_ii + 1
        
    for ii in indexes_lds:
        new_nodes[new_ii,:] = nodes[ii,:]
        new_indexes_lds.append(new_ii)
        indexes_translate[ii] = new_ii
                
        new_ii = new_ii + 1
        
    for ll in range(len(lines)):
        ii = int(lines[ll,0])
        jj = int(lines[ll,1])
        dd = lines[ll,2]
        
        new_lines[ll,:] = [indexes_translate[ii], indexes_translate[jj], dd ]
    
    return new_nodes, new_lines, new_indexes_reg, new_indexes_sol, new_indexes_hid, new_indexes_lds


#############################################################
def numpy_to_wide_latex_table(arr, caption=None, label=None, precision=4):
    """
    Convert a NumPy array into a wide LaTeX table with:
      - an index column
      - 3 repeated blocks across the page
      - each block contains: Index | Col1 | Col2 | Complex

    Parameters
    ----------
    arr : np.ndarray
        Shape (N, 3)
        First two columns: integers
        Third column: complex numbers
        
    Function by chatgpt
    """

    nrows = len(arr)

    # Split rows into 3 vertical sections
    chunk_size = math.ceil(nrows / 3)

    chunks = [
        arr[i:i + chunk_size]
        for i in range(0, nrows, chunk_size)
    ]

    # Ensure exactly 3 chunks
    while len(chunks) < 3:
        chunks.append([])

    lines = []

    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    
    if caption:
        lines.append(rf"\caption{{{caption}}}")

    if label:
        lines.append(rf"\label{{{label}}}")
    
    lines.append(r"\resizebox{1.00\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{c c c c | c c c c | c c c c}")
    lines.append(r"\hline")

    # Header
    header = (
        r"Line & $i$ & $j$ & $Z_{ij}$ & "
        r"Line & $i$ & $j$ & $Z_{ij}$ & "
        r"Line & $i$ & $j$ & $Z_{ij}$ \\"
    )
    lines.append(header)

    lines.append(r"\hline")

    # Maximum rows among chunks
    max_rows = max(len(c) for c in chunks)

    for r in range(max_rows):

        row_entries = []

        for chunk_id, chunk in enumerate(chunks):

            if r < len(chunk):

                global_index = chunk_id * chunk_size + r + 1

                a = int(chunk[r][0])
                b = int(chunk[r][1])

                z = complex(chunk[r][2])

                real = round(z.real, precision)
                imag = round(z.imag, precision)

                if imag >= 0:
                    z_str = f"${real}+{imag}\jhat$"
                else:
                    z_str = f"${real}-{abs(imag)}\jhat$"

                row_entries.extend([
                    str(global_index),
                    str(a),
                    str(b),
                    z_str
                ])

            else:
                # Empty cells if chunk shorter
                row_entries.extend(["", "", "", ""])

        lines.append(" & ".join(row_entries) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")

    lines.append(r"\end{table*}")

    return "\n".join(lines)

#############################################################
def nodes_ratings_table(
    vector,
    indexes_1,
    indexes_2,
    indexes_3,
    headers,
    precision=5,
    caption=None,
    label=None
):
    """
    Create a LaTeX table with 3 horizontal chunks.

    Each chunk has:
        index | corresponding floating-point value

    Parameters
    ----------
    vector : np.ndarray
        Vector containing floating-point values.

    indexes_1, indexes_2, indexes_3 : iterable
        Index sets.

    Returns
    -------
    str
        LaTeX table code.
        
    Function by chatgpt
    """

    chunks = [indexes_1, indexes_2, indexes_3]

    max_len = max(len(c) for c in chunks)

    lines = []

    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    if caption:
        lines.append(rf"\caption{{{caption}}}")

    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{c c | c c | c c}")
    lines.append(r"\hline")

    # Header
    lines.append(
        rf"{headers[0]} & {headers[1]} & "
        rf"{headers[2]} & {headers[3]} & "
        rf"{headers[4]} & {headers[5]} \\"
    )

    lines.append(r"\hline")

    for i in range(max_len):

        row_entries = []

        for chunk in chunks:

            if i < len(chunk):

                idx = int(chunk[i])
                value = float(np.asarray(vector[idx]).squeeze())

                row_entries.extend([
                    str(idx),
                    f"{value:.{precision}f}"
                ])

            else:
                row_entries.extend(["", ""])

        lines.append(" & ".join(row_entries) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")


    lines.append(r"\end{table*}")

    return "\n".join(lines)

#############################################################
def nodes_ratings_table_horizontal(
    vector,
    indexes_1,
    indexes_2,
    indexes_3,
    headers,
    precision=5,
    caption=None,
    label=None
):
    """
    Create a LaTeX table where each chunk is horizontal.

    Structure:

        Header1 | idx1 idx2 idx3 ...
        Value1  | val1 val2 val3 ...

        Header2 | ...
        Value2  | ...

        Header3 | ...
        Value3  | ...

    Parameters
    ----------
    vector : np.ndarray
        Vector containing floating-point values.

    indexes_1, indexes_2, indexes_3 : iterable
        Index sets.

    headers : list[str]
        Must contain 6 strings:
            [idx_header1, value_header1,
             idx_header2, value_header2,
             idx_header3, value_header3]
    """

    chunks = [indexes_1, indexes_2, indexes_3]

    # Largest chunk determines width
    max_len = max(len(c) for c in chunks)

    # Table format:
    # first column for labels + data columns
    col_format = "c | " + " ".join(["c"] * max_len)

    lines = []

    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")

    if caption:
        lines.append(rf"\caption{{{caption}}}")

    if label:
        lines.append(rf"\label{{{label}}}")

    lines.append(r"\resizebox{1.00\columnwidth}{!}{")
    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\hline")

    # =========================================================
    # Build each chunk horizontally
    # =========================================================

    for k, chunk in enumerate(chunks):

        idx_header = headers[2*k]
        val_header = headers[2*k + 1]

        # ---------------------------
        # Index row
        # ---------------------------

        idx_row = [idx_header]

        for idx in chunk:
            idx_row.append(str(int(idx)))

        # Fill remaining empty cells
        idx_row.extend([""] * (max_len - len(chunk)))

        lines.append(" & ".join(idx_row) + r" \\")

        # ---------------------------
        # Value row
        # ---------------------------

        val_row = [val_header]

        for idx in chunk:

            value = float(np.asarray(vector[idx]).squeeze())

            val_row.append(
                f"{value:.{precision}f}"
            )

        val_row.extend([""] * (max_len - len(chunk)))

        lines.append(" & ".join(val_row) + r" \\")

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)
#############################################################
def performance_table(
    data,
    row_names,
    column_headers,
    precisions,
    caption=None,
    label=None
):
    """
    Generate a LaTeX table with:
        - header row
        - first column = row names
        - remaining columns = floating-point values
        - individual precision for each column

    Parameters
    ----------
    data : np.ndarray
        Shape (3, N)

    row_names : list[str]
        Names of the 3 rows.

    column_headers : list[str]
        Headers for columns 2..N+1.

    precisions : list[int]
        Decimal precision for each data column.

    Returns
    -------
    str
        LaTeX table code.
        
    Function by chatgpt
    """

    data = np.asarray(data)

    if data.shape[0] != 3:
        raise ValueError("data must have exactly 3 rows")

    ncols = data.shape[1]

    if len(row_names) != 3:
        raise ValueError("row_names must contain exactly 3 strings")

    if len(column_headers) != ncols:
        raise ValueError(
            "column_headers length must match number of columns"
        )

    if len(precisions) != ncols:
        raise ValueError(
            "precisions length must match number of columns"
        )


    # Table format:
    # first column left aligned, rest centered
    col_format = "l " + " ".join(["c"] * ncols)

    lines = []

    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    
    if caption:
        lines.append(rf"\caption{{{caption}}}")

    if label:
        lines.append(rf"\label{{{label}}}")

    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\hline")

    # Header row
    header_entries = [""] + list(column_headers)

    lines.append(
        " & ".join(header_entries) + r" \\"
    )

    lines.append(r"\hline")

    # Data rows
    for i in range(3):

        row_entries = [row_names[i]]

        for j in range(ncols):

            value = float(data[i, j])
            prec = precisions[j]

            row_entries.append(
                f"{value:.{prec}f}"
            )

        lines.append(
            " & ".join(row_entries) + r" \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    lines.append(r"\end{table}")

    return "\n".join(lines)