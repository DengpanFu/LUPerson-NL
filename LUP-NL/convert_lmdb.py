#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-03-20 14:32:29
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np
import cv2
import lmdb
import pickle
import argparse


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='convert extracted LUP-NL images to LMDB data.')
    parser.add_argument('-d',  '--base_imgs_dir',  type=str,  default='lup-nl')
    parser.add_argument('-s',  '--save_dir',       type=str,  default='lupnl_lmdb')
    parser.add_argument('-c',  '--check',          action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    lmdb_dir = os.path.join(args.save_dir, 'lmdb')
    key_path = os.path.join(args.save_dir, 'keys.pkl')
    base_dir = args.base_imgs_dir
    if not os.path.exists(lmdb_dir): os.makedirs(lmdb_dir)
    if not args.check:
        keys = []
        vnames = []
        pids = []
        env = lmdb.open(lmdb_dir, map_size=1e12)
        txn = env.begin(write=True)
        cnt = 0
        vid_cnt = 0
        countries = sorted(os.listdir(base_dir))
        for i, country in enumerate(countries):
            city_dir = os.path.join(base_dir, country)
            citys = sorted(os.listdir(city_dir))
            for j, city in enumerate(citys):
                vid_dir = os.path.join(city_dir, city)
                vids = sorted(os.listdir(vid_dir))
                for k, vid in enumerate(vids):
                    if vid in vnames: continue
                    vnames.append(vid)
                    vid_cnt += 1
                    key_vid = '{:05d}'.format(vid_cnt)
                    im_dir = os.path.join(vid_dir, vid)
                    names = sorted([x for x in os.listdir(im_dir) if x.endswith('jpg')])
                    for m, name in enumerate(names):
                        if cnt % 2000 == 0:
                            print('[{:3d}|{:3d}] country={:s}, [{:d}|{:d}] city={:s}, ' \
                                '[{:3d}|{:3d}] vid={:s}, [{:d}|{:d}] name={:s}'.format(i, 
                                    len(countries), country, j, len(citys), city, k, len(vids), 
                                    vid, m, len(names), name))
                        key = key_vid + '_' + name[:-4]
                        pid = int(name.split('_')[0])
                        im_path = os.path.join(im_dir, name)
                        with open(im_path, 'rb') as f:
                            im_str = f.read()
                        im = np.fromstring(im_str, np.uint8)
                        keys.append(key)
                        pids.append(pid)
                        key_byte = key.encode('ascii')
                        txn.put(key_byte, im)
                        cnt += 1
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.close()

        with open(key_path, 'wb') as f:
            pickle.dump({"keys": keys, "pids":pids}, f)
    else:
        assert(os.path.exists(lmdb_dir)), 'lmdb file: {} does not exist'.format(lmdb_dir)
        assert(os.path.exists(key_path)), 'key file: {} does not exist'.format(key_path)
        env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
        keys = pickle.load(open(key_path, "rb"))['keys'][:100]  # check and visulize the first 100 images.
        for key in keys:
            with env.begin(write=False) as txn:
                buf = txn.get(key.encode('ascii'))
            img_flat = np.frombuffer(buf, dtype=np.uint8)
            im = cv2.imdecode(img_flat, 1)
            cv2.imwrite('tmp_vis.jpg', im)
