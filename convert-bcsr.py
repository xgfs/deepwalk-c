import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.io import loadmat, savemat
from scipy.sparse import dok_matrix
from collections import defaultdict
from struct import pack
from sys import exit


def mat_to_bcsr(fname, mat):
    with open(fname, 'wb') as outf:
        nv = mat.shape[0]
        ne = mat.nnz
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


def process(args):
    if args.format == "mat":
        mtx = loadmat(args.input)[args.matfile_variable_name].tocsr()
        if args.undirected:
            mtx = mtx + mtx.T  # we dont care about weights anyway
        mat_to_bcsr(args.output, mtx)
    elif args.format in ['edgelist', 'adjlist']:
        pass
    else:
        raise Exception(
            "Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)
    nodes = set()
    with open(args.input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            if args.sep is None:
                splt = line.split()
            else:
                splt = line.split(args.sep)
            if args.format == "edgelist" and len(splt) > 2:
                if abs(float(splt[2]) - 1) >= 1e-4:
                    raise ValueError("Weighted graphs are not supported")
            for node in splt:
                nodes.add(node)
    number_of_nodes = len(nodes)
    isnumbers = is_numbers_only(nodes)
    if isnumbers:
        node2id = dict(zip(sorted(map(int, nodes)), range(number_of_nodes)))
    else:
        node2id = dict(zip(nodes), range(number_of_nodes))
    graph = defaultdict(set)
    with open(args.input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            if args.sep is None:
                splt = line.split()
            else:
                splt = line.split(args.sep)
            if isnumbers:
                src = node2id[int(splt[0])]
            else:
                src = node2id[splt[0]]
            for node in splt[1:]:
                if isnumbers:
                    tgt = node2id[int(node)]
                else:
                    tgt = node2id[node]
                graph[src].add(tgt)
                if args.undirected:
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
    with open(args.output, 'wb') as outf:
        outf.write(str.encode('XGFS'))
        outf.write(pack('q', number_of_nodes))
        outf.write(pack('q', number_of_edges))
        outf.write(pack(str(number_of_nodes) + 'i', *indptr[:-1]))
        outf.write(pack(str(number_of_edges) + 'i', *indices))


def main():
    parser = ArgumentParser("convert-bcsr",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--format', default='edgelist',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--sep', default=' ',
                        help='Separator of input file')

    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    exit(main())
