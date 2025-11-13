import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
from glob import glob
import os
import pandas as pd
from ast import literal_eval
import cv2

class GCPSelectorDEM:
    def __init__(self, img_file, dem_file, K, rvec, tvec, max_steps=1000, step_size=0.5):
        """
        img_file: grayscale image path
        dem_file: raster DEM path
        K: camera intrinsic matrix
        rvec: rotation vector of camera
        tvec: translation vector of camera
        max_steps: max ray steps along viewing ray
        step_size: step size in world units
        """
        self.img = plt.imread(img_file)
        if self.img.ndim == 3:
            self.img = np.mean(self.img, axis=2)
        
        with rasterio.open(dem_file) as f:
            self.dem = f
            dem_data = f.read(1)
            dem_data[dem_data==-9999] = np.nan
            self.dem_data = dem_data
        self.transform = self.dem.transform
        self.nodata = self.dem.nodata

        self.K = K
        self.R, _ = cv2.Rodrigues(rvec)
        self.tvec = tvec.reshape(3,1)
        self.C = -self.R.T @ self.tvec

        self.max_steps = max_steps
        self.step_size = step_size
        self.gcp_list = []

        # Setup plot
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, cmap='gray')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        print("Click on the image to select GCPs. Close the window when done.")

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        u, v = event.xdata, event.ydata
        X,Y,Z = self.ray_intersect_dem(u,v)
        if X is None:
            print("No intersection found with DEM.")
            return

        print(f"Clicked pixel ({u:.1f},{v:.1f}) -> world ({X:.2f},{Y:.2f},{Z:.2f})")
        self.gcp_list.append((u, v, X, Y, Z))
        self.ax.plot(u,v,'ro')
        self.fig.canvas.draw()

    def ray_intersect_dem(self, u, v):
        """
        Cast a ray from camera through pixel (u,v) and intersect with DEM.
        Returns (X,Y,Z) world coordinates or (None,None,None) if no intersection.
        """
        # 1. Direction in camera coordinates
        pixel = np.array([[u],[v],[1.0]])
        ray_dir_cam = np.linalg.inv(self.K) @ pixel
        ray_dir_cam = ray_dir_cam.flatten() / np.linalg.norm(ray_dir_cam)

        # 2. Direction in world coordinates
        ray_dir_world = self.R.T @ ray_dir_cam

        # 3. Ray marching
        C = self.C.flatten()             # camera center
        ray_dir_world = (self.R.T @ ray_dir_cam).flatten()  # direction vector

        for step in range(self.max_steps):
            pos = C + step*self.step_size*ray_dir_world
            pos = np.asarray(pos).flatten()
            if pos.size != 3:
                continue
            X, Y, Z_guess = pos

            # DEM intersection
            try:
                row, col = self.dem.index(X,Y)
                if not (0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]):
                    continue
                Z_dem = self.dem_data[row, col]
                if Z_dem == self.nodata:
                    continue
            except:
                continue

            if pos[2] <= Z_dem:  # hit DEM
                return X,Y,Z_dem
        return None,None,None

    def save(self, out_csv):
        if len(self.gcp_list) == 0:
            print("No GCPs to save")
            return
        df = pd.DataFrame(self.gcp_list, columns=['col_sample','row_sample','X','Y','Z'])
        df.to_csv(out_csv,index=False)
        print(f"Saved {len(self.gcp_list)} GCPs to {out_csv}")


ch = "ch01"
data_path = '/Users/rdcrlrka/Research/Soo_locks'
img_file = glob(os.path.join(data_path, 'camera_calibration', 'single_band_images', f'*{ch}*.tiff'))[0]
dem_file = os.path.join(data_path, 'inputs', 'lidar_DSM_filled_cropped.tif')
calib_file = os.path.join(data_path, 'camera_calibration', 'calibration_params', 'calibration_parameters_merged.csv')
gcp_out_csv = os.path.join(data_path, 'inputs', 'gcp', f"GCP_reflectors_{ch}.csv")

calib = pd.read_csv(calib_file)
for k in ['K', 'D', 'K_full', 'R_rectified', 't_rectified']:
    calib[k] = calib[k].apply(literal_eval)
calib_row = calib.loc[calib['channel']==ch]
K = np.array(calib_row['K'].values[0], dtype=np.float32)
rvec = np.array(calib_row['R_rectified'].values[0], dtype=np.float32).flatten()
R = cv2.Rodrigues(rvec)[0]
tvec = np.array(calib_row['t_rectified'].values[0])


selector = GCPSelectorDEM(img_file, dem_file, K, R, tvec)
plt.show()

# After closing the window
selector.save(gcp_out_csv)