import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

class Patcher:
    def __init__(self, img, patch_size):
        self.src_img = img
        self.img_size = img.shape
        self.patch_size = patch_size
        self.patch_h, self.patch_w, self.patch_d = patch_size

    def _cal_num_patches(self):
        patch_num = (int(self.img_size[0] / self.patch_size[0]),
                    int(self.img_size[1] / self.patch_size[1]))
        return patch_num

    def patch_map_size(self):
        return self._cal_num_patches()

    def get_patches(self):
        patches = []
        patch_num_h, patch_num_w = self._cal_num_patches()
        flat_patch_h, flat_patch_w = self.patch_h * self.patch_d, self.patch_w * self.patch_d
        for i in range(patch_num_h):
            for j in range(patch_num_w):
                i_st, i_ed = i * self.patch_h, (i + 1) * self.patch_h
                j_st, j_ed = j * self.patch_w, (j + 1) * self.patch_w
                patches.append(self.src_img[i_st:i_ed, j_st:j_ed].reshape(flat_patch_h, flat_patch_w))

        return patches

class ContrastEntropyMap:
    def __init__(self, patcher):
        self.patcher = patcher
        self.src_img = patcher.src_img
    
    def _calc_shannon_entropy(self, img):
        histogram, bin_edges = np.histogram(img, np.arange(255, dtype=np.float32), density=True)
        histogram_length = sum(histogram)
        samples_probability = [float(h) / histogram_length for h in histogram]

        # return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
        return -sum([p * math.log(p, 2) for p in histogram if p != 0])

    def get_ctrast_ent_map(self):
        map_h, map_w = self.patcher.patch_map_size()
        ctrast_ent_map = np.zeros((map_h, map_w), dtype = np.float32)
        patches = self.patcher.get_patches()
        for i in range(map_h):
            for j in range(map_w):
                entropy = self._calc_shannon_entropy(
                    patches[i * map_w + j]
                )
                ctrast_ent_map[i][j] = entropy
                
        ctrast_ent_map = ctrast_ent_map / (np.max(ctrast_ent_map) + 0.00001)
        return ctrast_ent_map
    
class EntropyMap3D:
    def __init__ (self, scene, patch_size):
        # [H, W, N_samples]
        self.scene = scene
        self.patch_size = patch_size
    
    def get_ent_map(self):
        patcher = Patcher(self.scene, self.patch_size)
        scene_ent_map = ContrastEntropyMap(patcher).get_ctrast_ent_map()
        return scene_ent_map

class EntropyHeatmap:
    def __init__(self, img):
        self.img = img
        self.img_h, self.img_w, self.img_channels = img.shape
    
    def overlay_heatmap_on_image(self, heatmap):
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)/255
        overlay = heatmap * 0.6 + np.float32(self.img)/255
        overlay = overlay / np.max(overlay)
        return np.uint8(255*overlay)
    
    def apply_heatmap(self, entropy_map):
        entropy_map_extended = cv2.resize(entropy_map, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        return self.overlay_heatmap_on_image(entropy_map_extended)
        