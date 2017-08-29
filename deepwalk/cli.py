#   encoding: utf8
#   cli.py
"""Converter for three common graph formats (MATLAB sparse matrix, adjacency
list, edge list) can be found in the root directory of the project.
"""

from collections import defaultdict
from scipy.io import loadmat
from struct import pack

import click
import numpy as np


def mat_to_bcsr(fname, mat):
    with open(fname, 'wb') as outf:
        nv = mat.shape[0]
        ne = mat.nnz
        print('nv', nv, 'ne', ne)
        outf.write(str.encode('XGFS'))
        outf.write(pack('q', nv))
        outf.write(pack('q', ne))
        outf.write(pack(str(nv) + 'i', *mat.indptr[:-1]))
        outf.write(pack(str(ne) + 'i', *mat.indices))


def is_numbers_only(nodes):
    for node in nodes:
        try:
            int(node)
        except ValueError:
            return False
    return True


def process(format, matfile_variable_name, undirected, sep, input, output):
    if format == "mat":
        mtx = loadmat(input)[matfile_variable_name].tocsr()
        if undirected:
            mtx = mtx + mtx.T  # we dont care about weights anyway
        mat_to_bcsr(output, mtx)
    elif format in ['edgelist', 'adjlist']:
        pass

    nodes = set()
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            line = line.strip()
            if sep is None:
                splt = line.split()
            else:
                splt = line.split(sep)
            if format == "edgelist":
                if len(splt) == 3:
                    if abs(float(splt[2]) - 1) >= 1e-4:
                        raise ValueError("Weighted graphs are not supported")
                    else:
                        splt = splt[:-1]
                else:
                    raise ValueError("Incorrect graph format")
            for node in splt:
                nodes.add(node)
    number_of_nodes = len(nodes)
    isnumbers = is_numbers_only(nodes)
    print('Node IDs are numbers: ', isnumbers)
    if isnumbers:
        node2id = dict(zip(sorted(map(int, nodes)), range(number_of_nodes)))
    else:
        node2id = dict(zip(nodes), range(number_of_nodes))
    graph = defaultdict(set)
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            line = line.strip()
            if sep is None:
                splt = line.split()
            else:
                splt = line.split(sep)
            if isnumbers:
                src = node2id[int(splt[0])]
            else:
                src = node2id[splt[0]]
            if format == "edgelist" and len(splt) == 3:
                splt = splt[:-1]
            for node in splt[1:]:
                if isnumbers:
                    tgt = node2id[int(node)]
                else:
                    tgt = node2id[node]
                graph[src].add(tgt)
                if undirected:
                    graph[tgt].add(src)
    indptr = np.zeros(number_of_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(number_of_nodes):
        indptr[i + 1] = indptr[i] + len(graph[i])
    number_of_edges = indptr[-1]
    indices = np.zeros(number_of_edges, dtype=np.int32)
    cur = 0
    for node in range(number_of_nodes):
        for adjv in sorted(graph[node]):
            indices[cur] = adjv
            cur += 1
    print('nv', number_of_nodes, 'ne', number_of_edges)
    with open(output, 'wb') as outf:
        outf.write(str.encode('XGFS'))
        outf.write(pack('q', number_of_nodes))
        outf.write(pack('q', number_of_edges))
        outf.write(pack(str(number_of_nodes) + 'i', *indptr[:-1]))
        outf.write(pack(str(number_of_edges) + 'i', *indices))


@click.command(help=__doc__)
@click.option('--format',
              default='edgelist',
              type=click.Choice(['mat', 'edgelist', 'adjlist']),
              help='File format of input file')
@click.option('--matfile-variable-name', default='network',
              help='variable name of adjacency matrix inside a .mat file.')
@click.option('--undirected/--directed', default=True, is_flag=True,
              help='Treat graph as undirected.')
@click.option('--sep', default=' ', help='Separator of input file')
@click.argument('input', type=click.Path())
@click.argument('output', type=click.Path())
def main(format, matfile_variable_name, undirected, sep, input, output):
    process(format, matfile_variable_name, undirected, sep, input, output)