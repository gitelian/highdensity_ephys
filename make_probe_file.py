
#### remap neuronexus contact IDs to the order
#### we want them (0: bottom, 32: top of probe)

# site:channel
s = {1: 0,
     2: 1,
     3: 2,
     4: 3,
     5: 4,
     6: 5,
     7: 6,
     8: 7,
     9: 8,
     10: 9,
     }

#### randomly generate dead channels

# from numpy import random
# from pprint import pprint

# for dead_site in random.choice(s.values(),5):
#     s[dead_site]=None

# pprint(s)

# define probe geometry via coordinates
channel_groups = {
    # Shank index.
    0: {
        # List of channels to keep for spike detection.
        'channels': s.values(),

        # 2D positions of the channels
        # channel: (x,y)
        'geometry': {
            s[9]: (0,200),
            s[7]: (0,150),
            s[5]: (0,100),
            s[3]: (0,50),
            s[1]: (0,0),
            }
    1: {
            s[32]: (50,775), # column 1
            s[30]: (50,725),
            s[28]: (50,675),
            s[26]: (50,625),
            s[24]: (50,575),
        }
    }
}
print(channel_groups)

#### strip dead channels (i.e. None) out

new_group = {}
for gr, group in channel_groups.iteritems():
    new_group[gr] = {
        'channels': [],
        'geometry': {}
    }
    new_group[gr]['channels'] = [ch for ch in group['channels'] if ch is not None]
    new_group[gr]['geometry'] = {ch:xy for (ch,xy) in group['geometry'].iteritems() if ch is not None}

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
