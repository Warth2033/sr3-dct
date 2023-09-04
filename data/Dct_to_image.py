from PIL import Image
import numpy as np
from scipy.fftpack import idctn


class DctToImage():
    def __init__(self, dct):
        r_resolution = dct.shape()[0] * 8
        def zigzag_unflatten(arr):
            n = 8  # 数组的维度
            matrix = np.zeros((n, n), dtype=np.float16)
            row, col = 0, 0
            going_up = True

            for num in arr:
                matrix[row][col] = num

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
                    if col > 0 and row < n - 1:
                        row += 1
                        col -= 1
                    else:
                        if row == n - 1:
                            col += 1
                        else:
                            row += 1
                        going_up = True
            return matrix

        # Function to perform inverse DCT
        def inverse_dct(dct_coeffs):
            img_block = idctn(dct_coeffs, type=2, norm='ortho')  # 应用IDCT 
            return img_block.astype(np.uint8)

        img_reconstructed = np.zeros((r_resolution, r_resolution, 3), dtype=np.uint8)

        def reconstruct_image(dct, img_reconstructed):
            for i in range(dct.shape[0]):
                for j in range(dct.shape[1]):
                    for c in range(3):  # 对于每个颜色通道
                        reshaped_block = np.zeros((1, 1, 64), dtype=np.float16)
                        reshaped_block = dct[i, j, c*64:(c+1)*64]
                        
                        block = np.zeros((8, 8), dtype=np.float16)
                        block = zigzag_unflatten(reshaped_block)
                        
                        # 执行逆DCT变换
                        img_block = np.zeros((8, 8), dtype=np.uint8)
                        img_block = inverse_dct(block)
                        
                        # 将8x8图像块存储到完整的图像中
                        img_reconstructed[i*8:(i+1)*8, j*8:(j+1)*8, c] = img_block
            return img_reconstructed

        # 重建图像
        img_reconstructed = reconstruct_image(dct, img_reconstructed)

        # 将Numpy数组转换为PIL图像
        img_reconstructed_pil = Image.fromarray(img_reconstructed, mode='YCbCr')
        return img_reconstructed_pil