from io import BytesIO
import lmdb
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random
import util as Util
import unittest
import torch
from scipy.fftpack import idctn


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=True)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                hr_dct_bytes = txn.get(
                    'hr_dct_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_dct_bytes = txn.get(
                    'sr_dct_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None) or (hr_dct_bytes is None) or (sr_dct_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    hr_dct_bytes = txn.get(
                        'hr_dct_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_dct_bytes = txn.get(
                        'sr_dct_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                dct_HR = np.frombuffer(hr_dct_bytes, dtype=np.float16).reshape(64 * 3, self.r_res // 8, self.r_res // 8)
                dct_SR = np.frombuffer(sr_dct_bytes, dtype=np.float16).reshape(64 * 3, self.r_res // 8, self.r_res // 8)
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': dct_HR, 'SR': dct_SR, 'Index': index}
            # return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': dct_HR, 'SR': dct_SR, 'Index': index}


# # 测试类
# class TestLRHRDataset(unittest.TestCase):
#     # 测试函数
#     def test_dataset(self):
#         # Path to your LMDB or image directory
#         dataroot = 'dataset/tk_train_test_16_128'
#         datatype = 'lmdb'  # or 'img'
#         l_resolution = 16
#         r_resolution = 128

#         # Create dataset instance
#         dataset = LRHRDataset(dataroot, datatype, l_resolution, r_resolution)

#         # 检查数据集的长度
#         self.assertEqual(len(dataset), dataset.data_len)

#         # 从数据集中获取一个项目
#         item = dataset[0]

#         # 检查该项目是否包含预期的键
#         self.assertIn('img_HR', item)
#         self.assertIn('img_SR', item)
#         self.assertIn('HR', item)
#         self.assertIn('SR', item)

#         # 检查图像的形状和类型
#         self.assertIsInstance(item['img_HR'], torch.Tensor)
#         self.assertIsInstance(item['img_SR'], torch.Tensor)
#         self.assertEqual(item['img_HR'].shape, (3, r_resolution, r_resolution))
#         self.assertEqual(item['img_SR'].shape, (3, r_resolution, r_resolution))

#         # 看看 DCT 系数的形状和类型
#         self.assertIsInstance(item['HR'], np.ndarray)
#         self.assertIsInstance(item['SR'], np.ndarray)
#         self.assertEqual(item['HR'].shape, ( 64 * 3, r_resolution // 8, r_resolution // 8))
#         self.assertEqual(item['SR'].shape, ( 64 * 3, r_resolution // 8, r_resolution // 8))

#         def zigzag_unflatten(arr):
#             # print("打印zigzag_unflatten前的DCT系数: ", arr,'\n\n')  # 打印传入的DCT系数
#             n = 8  # 数组的维度
#             matrix = np.zeros((n, n), dtype=np.float16)
#             row, col = 0, 0
#             going_up = True

#             for num in arr:
#                 matrix[row][col] = num

#                 # 向上遍历
#                 if going_up:
#                     if row > 0 and col < n - 1:
#                         row -= 1
#                         col += 1
#                     else:
#                         if col == n - 1:
#                             row += 1
#                         else:
#                             col += 1
#                         going_up = False

#                 # 向下遍历
#                 else:
#                     if col > 0 and row < n - 1:
#                         row += 1
#                         col -= 1
#                     else:
#                         if row == n - 1:
#                             col += 1
#                         else:
#                             row += 1
#                         going_up = True
#             # print("打印zigzag_unflatten后的DCT系数: ", matrix,'\n\n')  # 打印传入的DCT系数
#             return matrix

#         # Function to perform inverse DCT
#         def inverse_dct(dct_coeffs):
#             # print("打印IDCT前的DCT系数: ", dct_coeffs,'\n\n')  # 打印传入的DCT系数
#             img_block = idctn(dct_coeffs, type=2, norm='ortho')  # 应用IDCT
#             # print("打印IDCT后的数据: ", img_block, '\n\n')  # 打印IDCT后的数据
#             # print("打印IDCT后的数据(.astype(np.uint8))): ", img_block.astype(np.uint8), '\n\n')  
#             # 打印IDCT后的数据
#             return img_block.astype(np.uint8)


#         # Get the DCT coefficients
#         dct_HR = item['HR'] # r/8*r/8*192
#         # np.savetxt('LRHR中间数据/hr-dct-reshaped(数组)（Ycbcr通道）.csv', dct_HR.reshape(-1,192), delimiter=',')#####################
#         dct_SR = item['SR']

#         img_HR_reconstructed = np.zeros((r_resolution, r_resolution, 3), dtype=np.uint8)
#         img_SR_reconstructed = np.zeros((r_resolution, r_resolution, 3), dtype=np.uint8)

#         def reconstruct_image(dct, img_reconstructed):
#             for c in range(3):
#                 for i in range(dct.shape[1]):
#                     for j in range(dct.shape[2]):  # 对于每个颜色通道
#                         reshaped_block = np.zeros((1, 1, 64), dtype=np.float16)
#                         reshaped_block = dct[c*64:(c+1)*64, i, j]
#                         # np.savetxt('LRHR中间数据/hr第一块dct-reshaped(数组)（Ycbcr通道）.csv', vector, delimiter=',')#####################
                        
#                         block = np.zeros((8, 8), dtype=np.float16)
#                         block = zigzag_unflatten(reshaped_block)
#                         # np.savetxt('LRHR中间数据/hr第一块dct(数组)（Ycbcr通道）.csv', reshaped_block, delimiter=',')#####################
                        
#                         # 执行逆DCT变换
#                         img_block = np.zeros((8, 8), dtype=np.uint8)
#                         img_block = inverse_dct(block)
#                         # np.savetxt('LRHR中间数据/hr第一块img(数组)（Ycbcr通道）.csv', img_block, delimiter=',')#####################
                        
#                         # 将8x8图像块存储到完整的图像中
#                         img_reconstructed[i*8:(i+1)*8, j*8:(j+1)*8, c] = img_block
#             return img_reconstructed

#         # 重建图像
#         img_HR_reconstructed = reconstruct_image(dct_HR, img_HR_reconstructed)
#         # np.savetxt('LRHR中间数据/hr(数组)（Ycbcr通道）.csv', img_HR_reconstructed.reshape(-1, 3), delimiter=',')#####################
#         img_SR_reconstructed = reconstruct_image(dct_SR, img_SR_reconstructed)

#         # 将Numpy数组转换为PIL图像
#         img_HR_reconstructed_pil = Image.fromarray(img_HR_reconstructed, mode='YCbCr')
#         # np.savetxt('LRHR中间数据/hr(Image)（Ycbcr通道）.csv', img_HR_reconstructed_pil.getdata(), delimiter=',')#####################
#         img_SR_reconstructed_pil = Image.fromarray(img_SR_reconstructed, mode='YCbCr')

#         # 保存为JPEG
#         img_HR_reconstructed_pil.save('HR_reconstructed.jpeg')
#         # img_HR_reconstructed_pil.show()
#         # np.savetxt('LRHR-hr（Image对象）（Ycbcr通道）.csv', list(img_HR_reconstructed_pil.getdata()), delimiter=',')
#         img_SR_reconstructed_pil.save('SR_reconstructed.jpeg')
#         # img_SR_reconstructed_pil.show()

# if __name__ == '__main__':
#     unittest.main()
