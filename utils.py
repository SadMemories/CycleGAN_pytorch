import random

import torch
import torch.nn as nn


# Discrimintor 图片缓存类
class ImagePool():

    def __init__(self, pool_size):
        self.pool_size = pool_size

        if self.pool_size > 0:
            self.images = []
            self.cur_cnt = 0

    def query(self, images):

        if self.pool_size == 0:
            return images

        ret_images = []
        for image in images:

            if self.cur_cnt < self.pool_size:
                ret_images.append(image)
                self.images.append(image)
                self.cur_cnt += 1
            else:
                p = random.uniform(0.0, 1.0)
                if p > 0.5:
                    index = random.randint(0, self.pool_size-1)
                    cur_img = self.images[index].clone()
                    self.images[index] = image
                    ret_images.append(cur_img)
                else:
                    ret_images.append(image)
        ret_images = torch.stack(ret_images, 0)
        return ret_images
