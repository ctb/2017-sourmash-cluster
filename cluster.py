import sys
import os
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
    parser.add_argument('sigs', nargs='+')
    parser.add_argument('--scaled', default=0, type=int)
    parser.add_argument('-q', '--quiet', action='store_true',
                   help='suppress non-error output')
    parser.add_argument('-d', '--debug', action='store_true',
                   help='output debugging output')
    parser.add_argument('--prefix', help='prefix for output files',
                        default='sourmash.clust')
    sourmash_args.add_ksize_arg(parser, sourmash_args.DEFAULT_LOAD_K)
    sourmash_args.add_moltype_args(parser)
    args = parser.parse_args()

    set_quiet(args.quiet, args.debug)

    if not check_files_exist(*args.sigs):
        sys.exit(-1)

    moltype = sourmash_args.calculate_moltype(args)

    # track ksizes, moltypes and error out early if we're not sure
    # which one to use.
    ksizes = set()
    moltypes = set()

    sigs = []
    for n, filename in enumerate(args.sigs):
        notify(u'\r\033[K', end=u'')
        notify('... loading file {} of {}', n + 1, len(args.sigs), end='\r')
        loaded = sig.load_signatures(filename, ksize=args.ksize,
                                     select_moltype=moltype)
        loaded = list(loaded)
        sigs += [ (x, filename) for x in loaded ]

        # track ksizes/moltypes
        for xs in loaded:
            ksizes.add(xs.minhash.ksize)
            moltypes.add(sourmash_args.get_moltype(xs))

        # error out while loading if we have more than one ksize/moltype
        if len(ksizes) > 1 or len(moltypes) > 1: break
    print('')

    # error exit?
    if len(ksizes) > 1 or len(moltypes) > 1:
        error("ksizes: {}", ksizes)
        error("moltypes: {}", moltypes)
        error("too many ksizes or molecule types, exiting.")
        sys.exit(-1)

    siglist = [ x for (x, _) in sigs ]

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

    notify("... comparing {} signatures, all by all", len(siglist))
    
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

            similarity = E.similarity(E2)
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
    for i in range(len(sigs)):
        label = '{}'.format(str(i))
        idx = i
        labels_to_sigs[label] = sigs[i]
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
    notify('** wrote clust dendrogram to {}', dendrogram_out)

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

    output_headers = ("cluster_id", "cluster_size", "filename", "name")
    output_rows = []
    for i in clusters:
        cluster_size = len(clusters[i])
        for x in clusters[i]:
            xs, sigfile = labels_to_sigs[x]

            row = (i, cluster_size, sigfile, xs.name())
            output_rows.append(row)
            

    csvfile = args.prefix + '.csv'
    with open(csvfile, 'wt') as fp:
        w = csv.writer(fp)
        w.writerow(output_headers)
        for row in output_rows:
            w.writerow(row)
    notify('** wrote clust assignments spreadsheet to {}', csvfile)

    # output clusters with more than one signature
    for i in clusters:
        if len(clusters[i]) == 1:
            continue

        cluster_dir = "{}.clust{}".format(args.prefix, i)
        os.mkdir(cluster_dir)

        for n, x in enumerate(clusters[i]):
            with open('{}/{}.sig'.format(cluster_dir, n), 'wt') as fp:
                xs, sigfile = labels_to_sigs[x]
                sig.save_signatures([ xs ], fp)

    # output singletons
    for i in clusters:
        if len(clusters[i]) == 1:
            output_name = '{}.singleton{}.sig'.format(args.prefix, i)

            x, = clusters[i]
            with open(output_name, 'wt') as fp:
                xs, sigfile = labels_to_sigs[x]
                sig.save_signatures([ xs ], fp)


if __name__ == '__main__':
    sys.exit(main())
