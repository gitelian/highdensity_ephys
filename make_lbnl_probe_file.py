# replace channel number with None if it is broken 
# do this for only channel_groups...code will automatically
# remove contacts from geometry dictionary

channel_groups = {0: {'channels': [0,
   1,
   2,
   3,
   4,
   5,
   6,
   7,
   8,
   9,
   10,
   11,
   12,
   13,
   14,
   15,
   16,
   17,
   18,
   19,
   20,
   21,
   22,
   23,
   24,
   25,
   26,
   27,
   28,
   29,
   30,
   31,
   32,
   33,
   34,
   35,
   36,
   37,
   38,
   39,
   40,
   41,
   42,
   43,
   44,
   45,
   46,
   47,
   48,
   49,
   50,
   51,
   52,
   53,
   54,
   55,
   56,
   57,
   58,
   59,
   60,
   61,
   62,
   63],
  'geometry': {0: (0, 0),
   1: (0, 25),
   2: (0, 50),
   3: (0, 75),
   4: (0, 100),
   5: (0, 125),
   6: (0, 150),
   7: (0, 175),
   8: (0, 200),
   9: (0, 225),
   10: (0, 250),
   11: (0, 275),
   12: (0, 300),
   13: (0, 325),
   14: (0, 350),
   15: (0, 375),
   16: (0, 400),
   17: (0, 425),
   18: (0, 450),
   19: (0, 475),
   20: (0, 500),
   21: (0, 525),
   22: (0, 550),
   23: (0, 575),
   24: (0, 600),
   25: (0, 625),
   26: (0, 650),
   27: (0, 675),
   28: (0, 700),
   29: (0, 725),
   30: (0, 750),
   31: (0, 775),
   32: (25, 0),
   33: (25, 25),
   34: (25, 50),
   35: (25, 75),
   36: (25, 100),
   37: (25, 125),
   38: (25, 150),
   39: (25, 175),
   40: (25, 200),
   41: (25, 225),
   42: (25, 250),
   43: (25, 275),
   44: (25, 300),
   45: (25, 325),
   46: (25, 350),
   47: (25, 375),
   48: (25, 400),
   49: (25, 425),
   50: (25, 450),
   51: (25, 475),
   52: (25, 500),
   53: (25, 525),
   54: (25, 550),
   55: (25, 575),
   56: (25, 600),
   57: (25, 625),
   58: (25, 650),
   59: (25, 675),
   60: (25, 700),
   61: (25, 725),
   62: (25, 750),
   63: (25, 775)}}}

print(channel_groups)

#### strip dead channels (i.e. None) out

new_group = {}
for gr, group in channel_groups.iteritems():
    new_group[gr] = {
        'channels': [],
        'geometry': {}
    }
    new_group[gr]['channels'] = [ch for ch in group['channels'] if ch is not None]
    # old way
    #new_group[gr]['geometry'] = {ch:xy for (ch,xy) in group['geometry'].iteritems() if ch is not None}

    # my updated way so I don't have to do the same work twice (i.e. enter None
    # for broken contacts in both channels and geometry dictionaries
    new_group[gr]['geometry'] = {ch:xy for (ch,xy) in group['geometry'].iteritems() if group['channels'][ch] is not None}

channel_groups = new_group
print(channel_groups)

#### build adjacency graph from the probe geometry

from scipy import spatial
from scipy.spatial.qhull import QhullError
def get_graph_from_geometry(geometry):

    # let's transform the geometry into lists of channel names and coordinates
    chans,coords = zip(*[(ch,xy) for ch,xy in geometry.iteritems()])

    # we'll perform the triangulation and extract the
    try:
        tri = spatial.Delaunay(coords)
    except QhullError:
        # oh no! we probably have a linear geometry.
        chans,coords = list(chans),list(coords)
        x,y = zip(*coords)
        # let's add a dummy channel and try again
        coords.append((max(x)+1,max(y)+1))
        tri = spatial.Delaunay(coords)

    # then build the list of edges from the triangulation
    indices, indptr = tri.vertex_neighbor_vertices
    edges = []
    for k in range(indices.shape[0]-1):
        for j in indptr[indices[k]:indices[k+1]]:
            try:
                edges.append((chans[k],chans[j]))
            except IndexError:
                # let's ignore anything connected to the dummy channel
                pass
    return edges


def build_geometries(channel_groups):
    for gr, group in channel_groups.iteritems():
        group['graph'] = get_graph_from_geometry(group['geometry'])
    return channel_groups

channel_groups = build_geometries(channel_groups)
print(channel_groups[0]['graph'])

#### plot the adjacency graph

# %pylab inline
import matplotlib.pyplot as plt
def plot_channel_groups(channel_groups):

    n_shanks = len(channel_groups)

    f,ax = plt.subplots(1,n_shanks,squeeze=False)
    for sh in range(n_shanks):
        coords = [xy for ch,xy in channel_groups[sh]['geometry'].iteritems()]
        x,y = zip(*coords)
        ax[sh,0].scatter(x,y,color='0.2')

        for pr in channel_groups[sh]['graph']:
            points = [channel_groups[sh]['geometry'][p] for p in pr]
            ax[sh,0].plot(*zip(*points),color='k',alpha=0.2)

        ax[sh,0].set_xlim(min(x)-10,max(x)+10)
        ax[sh,0].set_ylim(min(y)-10,max(y)+10)
        ax[sh,0].set_xticks([])
        ax[sh,0].set_yticks([])
        ax[sh,0].set_title('group %i'%sh)

        plt.axis('equal')
        plt.show()

plot_channel_groups(channel_groups)
