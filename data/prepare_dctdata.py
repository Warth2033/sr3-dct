import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time
from scipy.fftpack import dctn


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img, mode):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()

def dct_convert_bytes(dct_coeffs_reshaped):
    return dct_coeffs_reshaped.tobytes()

def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False, mode='RGB'):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)
    # print("resize_multiple resize后hr图像像素值：", list(hr_img.getdata()))
    # 计算DCT系数并进行Zigzag重新排列
    def compute_dct_reshaped(image):
        if mode=='YCbCr':
            if image.mode != 'YCbCr':
                image = image.convert('YCbCr')
            # np.savetxt('prepare中间数据/hr（Image对象）（Ycbcr通道）.csv', image.getdata(), delimiter=',')#####################
        # image = Image.open(BytesIO(image))  # 从字节串创建图像对象
        img_array = np.array(image, dtype=np.uint8)
        # np.savetxt('prepare中间数据/hr（数组）（Ycbcr通道）.csv', img_array.reshape(-1, 3), delimiter=',')#####################
        dct_coeffs_reshaped = np.zeros((64 * 3, img_array.shape[0] // 8, img_array.shape[1] // 8), dtype=np.float32)
        for c in range(3):  # 对每个通道进行处理
            for i in range(0, img_array.shape[0], 8):
                for j in range(0, img_array.shape[1], 8):
                    block = img_array[i:i+8, j:j+8, c]
                    # if(i==0 and j==0 and c==0):
                    #     np.savetxt('prepare中间数据/hr第一块img（数组）.csv', block, delimiter=',')###################################
                    dct_coeffs = dctn(block, type=2, norm='ortho')
                    # if(i==0 and j==0 and c==0):
                    #     np.savetxt('prepare中间数据/hr第一块dct（数组）.csv', dct_coeffs, delimiter=',')###################################
                    dct_coeffs_reshaped[ c*64:(c+1)*64, i//8, j//8] = zigzag_flatten(dct_coeffs)
                    # if(i==0 and j==0 and c==0):
                    #     np.savetxt('prepare中间数据/hr第一块dct-reshaped（数组）.csv', dct_coeffs_reshaped[0][0], delimiter=',')###################################
        return dct_coeffs_reshaped
    if lmdb_save:
        hr_dct_reshaped = compute_dct_reshaped(hr_img)
        # np.savetxt('prepare中间数据/hr_dct_reshaped（数组）.csv', hr_dct_reshaped[0][0][0:64], delimiter=',')###################################
        sr_dct_reshaped = compute_dct_reshaped(sr_img)
        hr_dct_reshaped = dct_convert_bytes(hr_dct_reshaped)
        sr_dct_reshaped = dct_convert_bytes(sr_dct_reshaped)
        lr_img = image_convert_bytes(lr_img, mode)
        hr_img = image_convert_bytes(hr_img, mode)
        sr_img = image_convert_bytes(sr_img, mode)
        return lr_img, hr_img, sr_img, hr_dct_reshaped, sr_dct_reshaped
    return lr_img, hr_img, sr_img

def zigzag_flatten(matrix):
    m, n = matrix.shape
    result = np.empty(m * n, dtype=np.float32)  # 创建结果数组

    row, col = 0, 0
    going_up = True
    i = 0

    while row < m and col < n:
        result[i] = matrix[row][col]
        i += 1

        # 向上遍历
        if going_up:
            if row > 0 and col < n - 1:
                row -= 1
                col += 1
            else:
                if col == n - 1:
                    row += 1
                else:
                    col += 1
                going_up = False

        # 向下遍历
        else:
            if col > 0 and row < m - 1:
                row += 1
                col -= 1
            else:
                if row == m - 1:
                    col += 1
                else:
                    row += 1
                going_up = True

    return result


def resize_worker(img_file, sizes, resample, lmdb_save=False, mode='RGB'):
    img = Image.open(img_file)
    img = img.convert('RGB')  # 转换到RGB色域
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save, mode=mode)
    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img , hr_dct, sr_dct = imgs
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
                txn.put('hr_dct_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_dct)
                txn.put('sr_dct_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_dct)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False, mode='RGB'):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save, mode=mode)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024**4*5, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img , hr_dct, sr_dct = imgs
            if not lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
                sr_img.save(
                    '{}/sr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], i.zfill(5)))
            else:
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
                    txn.put('hr_dct_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_dct)
                    txn.put('sr_dct_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_dct)
            total += 1
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='{}/Dataset/celebahq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/celebahq')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')
    parser.add_argument('--mode', '-m', type=str, default='RGB')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb, mode=args.mode)
