#! /usr/bin/env python
"""
Clustering and cluster extraction on collections of sourmash signatures.

The current implementation is the dumbest possible approach: greedy
clustering + cluster agglomeration, such that cluster membership is
determined transitively by similarity.  That is, a cluster will contain
all signatures that are above the provided similarity threshold with
any other signature in the cluster.

Note that the output (if you use --save-dir) will be the *entire* signature
file copied into the cluster directories, so you can cluster at k=21 and
then have access to signatures of the clustered samples at k=51.

Usage:

   sourmash_cluster -k 31 --similarity=0.6 --save-dir clusters_out/

"""
import sys
import sourmash_lib
import collections
import argparse
import shutil
import os


def find_within(sig, sigset, min_similarity):
    x = set()
    for other in sigset:
        if sig.similarity(other) >= min_similarity:
            x.add(other)

    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sigs', nargs='+')
    p.add_argument('-k', '--ksize', default=31, type=int)
    p.add_argument('--similarity', default=0.5, type=float)
    p.add_argument('--save-dir', default=None)
    p.add_argument('--rm', action='store_true')
    args = p.parse_args()

    if args.save_dir:
        if os.path.exists(args.save_dir):
            if args.rm:
                print('removing {} because of --rm'.format(args.save_dir))
                shutil.rmtree(args.save_dir)
            else:
                print('{} already exists'.format(args.save_dir))
                sys.exit(-1)

        print('clusters will be placed in \'{}\''.format(args.save_dir))
        os.mkdir(args.save_dir)

    md5_to_filename = {}
    sigset = set()
    for filename in args.sigs:
        print("\r\033[Kloading {}".format(filename), end="")
        sig = sourmash_lib.load_one_signature(filename,
                                              select_ksize=args.ksize)
        md5_to_filename[sig.md5sum()] = filename
        sigset.add(sig)

    print('\nclustering...')

    clusters = collections.defaultdict(set)
    cluster_n = 0

    while sigset:
        sig = sigset.pop()
        
        cluster = set()
        cluster.add(sig)
        friends = find_within(sig, sigset, args.similarity)
        
        while friends:
            sigset -= friends

            new_friends = set()
            for f in friends:
                new_friends.update(find_within(f, sigset, args.similarity))
                sigset -= new_friends

            cluster.update(friends)
            friends = new_friends

        clusters[cluster_n] = cluster
        cluster_n += 1

    print('...done!')

    cluster_sizes = []
    singletons = []
    for k, members in clusters.items():
        if len(members) > 1:
            cluster_sizes.append((len(members), k))
        else:
            singletons.append(k)
    cluster_sizes.sort(reverse=True)

    print('{} signatures, {} clusters, {} singletons'.format(len(md5_to_filename), len(cluster_sizes), len(singletons)))
    
    for n, (_, k) in enumerate(cluster_sizes):
        print('cluster {} contains {} signatures'.format(n, len(clusters[k])))

        if args.save_dir:
            dirpath = os.path.join(args.save_dir, 'cluster{}'.format(n))
            os.mkdir(dirpath)

        for sig in clusters[k]:
            md5 = sig.md5sum()
            filename = md5_to_filename[md5]
            print('\t', filename)

            if args.save_dir:
                copyto = os.path.join(dirpath, filename)
                shutil.copyfile(filename, copyto)

    if args.save_dir and singletons:
        dirpath = os.path.join(args.save_dir, 'singletons')
        os.mkdir(dirpath)

        for k in singletons:
            sig = next(iter(clusters[k]))
            filename = md5_to_filename[sig.md5sum()]
            copyto = os.path.join(dirpath, filename)
            shutil.copyfile(filename, copyto)


if __name__ == '__main__':
    main()
