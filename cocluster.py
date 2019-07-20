import sys
import csv
import argparse
from sourmash import sourmash_args
from sourmash.lca.lca_utils import check_files_exist
from sourmash.logging import notify, error, debug, set_quiet, print_results
from sourmash.sourmash_args import SourmashArgumentParser
from sourmash import signature as sig
import collections


def main():
    import numpy
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab
    import scipy.cluster.hierarchy as sch
    parser = argparse.ArgumentParser()
    parser.add_argument('--cut-point', type=float, default=0.1)
    parser.add_argument('--first', nargs='+', action='append')
    parser.add_argument('--second', nargs='+', action='append')
    parser.add_argument('-q', '--quiet', action='store_true',
                   help='suppress non-error output')
    parser.add_argument('-d', '--debug', action='store_true',
                   help='output debugging output')
    parser.add_argument('-o', '--output')
    parser.add_argument('--ignore-abundance', action='store_true',
                        help='do NOT use k-mer abundances if present')
    parser.add_argument('--prefix', help='prefix for output files',
                        default='sourmash.coclust')
    sourmash_args.add_ksize_arg(parser, sourmash_args.DEFAULT_LOAD_K)
    sourmash_args.add_moltype_args(parser)
    args = parser.parse_args()

    if not (args.first and args.second):
        error('Error! must specify --first and --second list of sigs to co-cluster!')
        sys.exit(-1)

    set_quiet(args.quiet, args.debug)

    # flatten --first and --second
    args.first = [item for sublist in args.first for item in sublist]
    args.second = [item for sublist in args.second for item in sublist]

    # have to have two calls as python < 3.5 can only have one expanded list
    if not check_files_exist(*args.first):
        sys.exit(-1)

    if not check_files_exist(*args.second):
        sys.exit(-1)

    notify('first list contains {} files; second list contains {} files.',
           len(args.first), len(args.second))
    
    moltype = sourmash_args.calculate_moltype(args)

    first_sigs = []
    for n, filename in enumerate(args.first):
        notify(u'\r\033[K', end=u'')
        notify('... loading file {} of {} for first list', n, len(args.first),
               end='\r')
        loaded = sig.load_signatures(filename, ksize=args.ksize,
                                     select_moltype=moltype)
        loaded = list(loaded)
        first_sigs += [ (x, filename) for x in loaded ]
    print('')

    second_sigs = []
    for n, filename in enumerate(args.second):
        notify(u'\r\033[K', end=u'')
        notify('... loading file {} of {} for second list', n,
               len(args.second), end='\r')
        
        loaded = sig.load_signatures(filename, ksize=args.ksize,
                                     select_moltype=moltype)
        second_sigs += [ (x, filename) for x in loaded ]
    print('')

    siglist = [ x for (x, _) in first_sigs + second_sigs ]

    ksizes = set()
    moltypes = set()

    # track ksizes/moltypes
    for s in siglist:
        ksizes.add(s.minhash.ksize)
        moltypes.add(sourmash_args.get_moltype(s))

    # error out while loading if we have more than one ksize/moltype
    if len(ksizes) > 1 or len(moltypes) > 1:
        error("ksizes: {}", ksizes)
        error("moltypes: {}", moltypes)
        error("too many ksizes or molecule types, existing.")
        sys.exit(-1)

    notify('ksize: {} / moltype: {}', ksizes.pop(), moltypes.pop())
    
    # check to make sure they're potentially compatible - either using
    # max_hash/scaled, or not.
    scaled_sigs = [s.minhash.max_hash for s in siglist]
    is_scaled = all(scaled_sigs)
    is_scaled_2 = any(scaled_sigs)

    # complain if it's not all one or the other
    if is_scaled != is_scaled_2:
        error('cannot mix scaled signatures with bounded signatures')
        sys.exit(-1)

    # if using --scaled, downsample appropriately
    if is_scaled:
        max_scaled = max(s.minhash.scaled for s in siglist)
        notify('downsampling to scaled value of {}'.format(max_scaled))
        for s in siglist:
            s.minhash = s.minhash.downsample_scaled(max_scaled)

    notify('first list contains {} signatures; second list contains {} signatures.',
           len(args.first), len(args.second))


    notify("...comparing {} signatures, all by all", len(siglist))
    
    # build the distance matrix
    D = numpy.zeros([len(siglist), len(siglist)])
    numpy.set_printoptions(precision=3, suppress=True)

    # do all-by-all calculation
    count = 0
    total_count = int(len(siglist) ** 2 / 2)
    for i, E in enumerate(siglist):
        for j, E2 in enumerate(siglist):
            if i < j:
                continue
            count += 1

            if count % 250 == 0:
                pcnt = count / total_count * 100.0
                notify(u'\r\033[K', end=u'')
                notify('... comparing {} of {} ({:.0f}%)', count, total_count,
                       pcnt, end='\r')

            similarity = E.similarity(E2, args.ignore_abundance)
            D[i][j] = similarity
            D[j][i] = similarity
    print('')

    if len(siglist) < 30:
        for i, E in enumerate(siglist):
            # for small matrices, pretty-print some output
            name_num = '{}-{}'.format(i, E.name())
            if len(name_num) > 20:
                name_num = name_num[:17] + '...'
            print_results('{:20s}\t{}'.format(name_num, D[i, :, ],))

    print_results('min similarity in matrix: {:.3f}', numpy.min(D))

    dendrogram_out = args.prefix + '.dendro.pdf'
    labeltext = []
    labels_to_sigs = {}
    labels_to_first = {}
    labels_to_second = {}
    len_first_sigs = len(first_sigs)
    for i in range(len(first_sigs)):
        label = 'A.{}'.format(str(i))
        idx = i
        labels_to_sigs[label] = first_sigs[i]
        labels_to_first[label] = idx
        labeltext.append(label)
    for i in range(len_first_sigs, len(siglist)):
        label = 'B.{}'.format(str(i))
        idx = i - len_first_sigs
        labels_to_sigs[label] = second_sigs[i - len_first_sigs]
        labels_to_second[label] = idx
        labeltext.append(label)

    def augmented_dendrogram(*args, **kwargs):

        ddata = sch.dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            import matplotlib.pyplot as plt
            for i, d in zip(ddata['icoord'], ddata['dcoord']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                plt.plot(x, y, 'ro')
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                             textcoords='offset points',
                             va='top', ha='center')

        return ddata

    fig = pylab.figure(figsize=(8,5))
    ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax1.set_xticks([])
    ax1.set_yticks([])
    Y = sch.linkage(D, method='single')
    Z = augmented_dendrogram(Y, orientation='top', no_labels=True)
    fig.savefig(dendrogram_out)
    notify('** wrote coclust dendrogram to {}', dendrogram_out)

    CUT_POINT=args.cut_point

    # redo load/clustering just for grins.
    cluster_ids = sch.fcluster(Y, t=CUT_POINT, criterion='distance')

    Z = augmented_dendrogram(Y, orientation='top', no_labels=True, labels=labeltext)

    # now, get leaves and leaf labels
    idx1 = Z['leaves']
    new_labels = Z['ivl']

    # build clusters => sets of hashes.        
    clusters = collections.defaultdict(set)

    for i, k in enumerate(idx1):
        cluster_id = cluster_ids[k]
        clusters[cluster_id].add(new_labels[i])

    for i in clusters:
        print('cluster {} is {} in size'.format(i, len(clusters[i])))
        for x in clusters[i]:
            (xs, sigfile) = labels_to_sigs[x]
            print('\t', xs.name())

    output_headers = ("cluster_id", "origin", "filename", "name")
    output_rows = []
    for i in clusters:
        for x in clusters[i]:
            xs, sigfile = labels_to_sigs[x]

            origin = None
            if x in labels_to_first:
                origin = "first"
            elif x in labels_to_second:
                origin = "second"
            assert origin
            row = (i, origin, sigfile, xs.name())
            output_rows.append(row)
            

    csvfile = args.prefix + '.csv'
    with open(csvfile, 'wt') as fp:
        w = csv.writer(fp)
        w.writerow(output_headers)
        for row in output_rows:
            w.writerow(row)
    notify('** wrote coclust assignments spreadsheet to {}', csvfile)


if __name__ == '__main__':
    sys.exit(main())
