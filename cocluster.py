import sys
import os
import shutil
import csv
import argparse
from sourmash import sourmash_args
from sourmash.lca.lca_utils import check_files_exist
from sourmash.logging import notify, error, debug, set_quiet, print_results
from sourmash.sourmash_args import SourmashArgumentParser
from sourmash import signature as sig
import collections

import sys
sys.setrecursionlimit(10000)


def renumber_clusters_by_size(clusters):
    # sort by cluster size
    cluster_sizes = [ (len(clusters[i]), i) for i in clusters ]
    cluster_sizes.sort(reverse=True)

    # build new clusters
    new_clusters = {}
    for new_cluster_id, (_, old_cluster_id) in enumerate(cluster_sizes):
        new_clusters[new_cluster_id] = clusters[old_cluster_id]

    return new_clusters


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
    parser.add_argument('--scaled', default=0, type=int)
    parser.add_argument('-q', '--quiet', action='store_true',
                   help='suppress non-error output')
    parser.add_argument('-d', '--debug', action='store_true',
                   help='output debugging output')
    parser.add_argument('--prefix', help='prefix for output files',
                        default='sourmash.coclust')
    parser.add_argument('--threshold', type=float, default=0,
                        help='minimum threshold in bp for similarity')
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

    # track ksizes, moltypes and error out early if we're not sure
    # which one to use.
    ksizes = set()
    moltypes = set()

    first_sigs = []
    for n, filename in enumerate(args.first):
        notify('... loading file {} of {} for first list', n + 1,
               len(args.first), end='\r')
        loaded = sig.load_signatures(filename, ksize=args.ksize,
                                     select_moltype=moltype)
        loaded = list(loaded)
        first_sigs += [ (x, filename) for x in loaded ]

        # track ksizes/moltypes
        for xs in loaded:
            ksizes.add(xs.minhash.ksize)
            moltypes.add(sourmash_args.get_moltype(xs))

        # error out while loading if we have more than one ksize/moltype
        if len(ksizes) > 1 or len(moltypes) > 1: break
    notify('loaded {} files & {} signatures for first list', n + 1,
           len(first_sigs))

    second_sigs = []
    for n, filename in enumerate(args.second):
        notify('... loading file {} of {} for second list', n + 1,
               len(args.second), end='\r')
        
        loaded = sig.load_signatures(filename, ksize=args.ksize,
                                     select_moltype=moltype)
        second_sigs += [ (x, filename) for x in loaded ]

        # track ksizes/moltypes
        for xs in loaded:
            ksizes.add(xs.minhash.ksize)
            moltypes.add(sourmash_args.get_moltype(xs))

        # error out while loading if we have more than one ksize/moltype
        if len(ksizes) > 1 or len(moltypes) > 1:
            break
    notify('loaded {} files & {} signatures for second list', n + 1,
           len(second_sigs))

    # error exit?
    if len(ksizes) > 1 or len(moltypes) > 1:
        error("ksizes: {}", ksizes)
        error("moltypes: {}", moltypes)
        error("too many ksizes or molecule types, exiting.")
        sys.exit(-1)

    siglist = [ x for (x, _) in first_sigs + second_sigs ]

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
        if args.scaled:
            max_scaled = args.scaled

        notify('downsampling to scaled value of {}'.format(max_scaled))
        for s in siglist:
            s.minhash = s.minhash.downsample_scaled(max_scaled)

    elif args.scaled:
        error('cannot specify --scaled with non-scaled signatures.')
        sys.exit(-1)

    ### done loading!

    # if scaled, try filter
    leftover_first_sigs = []
    leftover_second_sigs = []
    if is_scaled and args.threshold:
        args.threshold = int(args.threshold)
        notify('filtering for at least {} shared k-mers between collections.',
               args.threshold)
        first_mins = set()
        for xs, _ in first_sigs:
            first_mins.update(xs.minhash.get_mins())

        second_mins = set()
        for xs, _ in second_sigs:
            second_mins.update(xs.minhash.get_mins())

        in_both = first_mins.intersection(second_mins)
        threshold = args.threshold / max_scaled

        # now filter
        def has_enough(xs):
            if len(in_both.intersection(xs.minhash.get_mins())) >= threshold:
                return 1
            return 0

        new_first_sigs = []
        leftover_first_sigs = []
        for xs, filename in first_sigs:
            if has_enough(xs):
                new_first_sigs.append((xs, filename))
            else:
                leftover_first_sigs.append((xs, filename))
        first_sigs = new_first_sigs

        new_second_sigs = []
        leftover_second_sigs = []
        for xs, filename in second_sigs:
            if has_enough(xs):
                new_second_sigs.append((xs, filename))
            else:
                leftover_second_sigs.append((xs, filename))
        second_sigs = new_second_sigs

        siglist = [ x for (x, _) in first_sigs + second_sigs ]

    notify('first list contains {} signatures; second list contains {} signatures.',
           len(args.first), len(args.second))


    notify("... comparing {} signatures, all by all.", len(siglist))
    
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
                notify('... comparing {} of {} ({:.0f}%)', count, total_count,
                       pcnt, end='\r')

            similarity = E.similarity(E2)
            D[i][j] = similarity
            D[j][i] = similarity
    notify('completed a total of {} comparisons.', total_count)

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
    for idx in range(len(first_sigs)):
        label = 'A.{}'.format(str(idx))
        labels_to_sigs[label] = first_sigs[idx]
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

    # build clusters => sets of samples
    clusters = collections.defaultdict(set)

    for i, k in enumerate(idx1):
        cluster_id = cluster_ids[k]
        clusters[cluster_id].add(new_labels[i])

    clusters = renumber_clusters_by_size(clusters)

    # add filtered out signatures, too; all have cluster size of 1.
    cluster_id = len(clusters)
    A_label = len(first_sigs)
    for xs, filename in leftover_first_sigs:
        label = 'A.{}'.format(str(A_label))
        clusters[cluster_id] = set([label])
        labels_to_sigs[label] = (xs, filename)
        labels_to_first[label] = A_label
        cluster_id += 1
        A_label += 1

    B_label = len(second_sigs)
    for xs, filename in leftover_second_sigs:
        label = 'B.{}'.format(str(B_label))
        clusters[cluster_id] = set([label])
        labels_to_sigs[label] = (xs, filename)
        labels_to_second[label] = B_label
        cluster_id += 1
        B_label += 1

    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) == 1:
            continue

        print_results('cluster {} is {} in size', i, len(cluster))
        for x in cluster:
            (xs, sigfile) = labels_to_sigs[x]
            print_results('\t{}', xs.name())

    # output a CSV summary
    output_headers = ("cluster_id", "cluster_size", "origin", "filename", "name")
    output_rows = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        cluster_size = len(cluster)
        for x in cluster:
            xs, sigfile = labels_to_sigs[x]

            origin = None
            if x in labels_to_first:
                origin = "first"
            elif x in labels_to_second:
                origin = "second"
            assert origin
            row = (i, cluster_size, origin, sigfile, xs.name())
            output_rows.append(row)

    csvfile = args.prefix + '.csv'
    with open(csvfile, 'wt') as fp:
        w = csv.writer(fp)
        w.writerow(output_headers)
        for row in output_rows:
            w.writerow(row)
    notify('** wrote coclust assignments spreadsheet to {}', csvfile)

    # output clusters with more than one signature into a directory
    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) == 1:
            continue

        cluster_dir = "{}.clust{}".format(args.prefix, i)
        try:
            shutil.rmtree(cluster_dir)
        except FileNotFoundError:
            pass
        os.mkdir(cluster_dir)

        for n, x in enumerate(cluster):
            with open('{}/{}.sig'.format(cluster_dir, x), 'wt') as fp:
                xs, sigfile = labels_to_sigs[x]
                sig.save_signatures([ xs ], fp)
    notify('** saved clusters to {}.clust*', args.prefix)

    # output singleton signatures to .sig files
    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) == 1:
            output_name = '{}.singleton{}.sig'.format(args.prefix, i)
            x, = cluster

            with open(output_name, 'wt') as fp:
                xs, sigfile = labels_to_sigs[x]
                sig.save_signatures([ xs ], fp)
    notify('** saved singletons to {}.singleton*.sig', args.prefix)

    # some summary status...
    n_pairs = 0
    n_singletons_A = 0
    n_singletons_B = 0
    n_pure_A = 0
    n_pure_B = 0
    n_multi_impure = 0

    for i in range(len(clusters)):
        cluster = clusters[i]
        cluster_size = len(cluster)
        origins = set()

        for x in cluster:
            xs, sigfile = labels_to_sigs[x]

            origin = None
            if x in labels_to_first:
                origin = "first"
            elif x in labels_to_second:
                origin = "second"
            assert origin
            origins.add(origin)

        if cluster_size == 1:
            if origin == 'first':
                n_singletons_A += 1
            elif origin == 'second':
                n_singletons_B += 1
        elif cluster_size == 2 and len(origins) == 2:
            n_pairs += 1
        elif len(origins) == 1:
            if origin == 'first':
                n_pure_A += 1
            elif origin == 'second':
                n_pure_B += 1
        else:
            n_multi_impure += 1

    print_results('total clusters: {}', len(clusters))
    print_results('num 1:1 pairs: {}', n_pairs)
    print_results('num singletons in first: {}', n_singletons_A)
    print_results('num singletons in second: {}', n_singletons_B)
    print_results('num multi-sig clusters w/only first: {}', n_pure_A)
    print_results('num multi-sig clusters w/only second: {}', n_pure_B)
    print_results('num multi-sig clusters mixed: {}', n_multi_impure)

    assert n_pairs + n_singletons_A + n_singletons_B + \
        n_pure_A + n_pure_B + n_multi_impure == len(clusters)


if __name__ == '__main__':
    sys.exit(main())
