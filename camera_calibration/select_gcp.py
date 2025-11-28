# !/usr/bin/env python

import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import xarray as xr


class PointPicker:
    def __init__(self, img_path, ortho_path, dem_path, cams_path, out_csv="match_points.csv"):

        # Load the input image
        self.img = rxr.open_rasterio(img_path).transpose("y", "x", "band").values

        # Load orthoimage
        ortho = rxr.open_rasterio(ortho_path).squeeze()
        ortho = xr.where(ortho==-9999, np.nan, ortho / 1e5)
        ortho = ortho.rio.write_crs("EPSG:32619")
        self.ortho = ortho

        # Load DEM and reproject it to match the ortho grid
        dem = rxr.open_rasterio(dem_path).squeeze()
        dem = xr.where(dem==-9999, np.nan, dem)
        dem = dem.rio.write_crs("EPSG:32619")
        self.dem = dem.rio.reproject_match(self.ortho)

        # Load camera positions
        cams = gpd.read_file(cams_path)
        cams = cams.to_crs("EPSG:32619")

        # Get the current camera position for better zooming on the orthoimage
        ch = int(os.path.basename(img_path).split('_')[0].split('ch')[1])
        cam_geom = cams.loc[cams['channel']==ch]['geometry'].values[0]
        self.ortho_xlim = cam_geom.coords.xy[0][0] - 10, cam_geom.coords.xy[0][0] + 10
        self.ortho_ylim = cam_geom.coords.xy[1][0] - 10, cam_geom.coords.xy[1][0] + 10

        # Ortho pixel grid info
        self.transform = self.ortho.rio.transform()

        # Internal storage
        self.points = []
        self.out_csv = out_csv
        self.mode = "image"
        self.pending_img_pt = None
        self.loaded_points = []

        # Plot setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            1, 2, figsize=(14, 8), 
            gridspec_kw=dict(width_ratios=[1.5,1])
            )

        # Set up mouse motion/clicking settings
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.onpress_pan)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.onpan)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.onrelease_pan)

        # Load prior points if CSV exists
        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            for _,row in df.iterrows():
                p = [
                    float(row["img_px"]),
                    float(row["img_py"]),
                    float(row["X"]),
                    float(row["Y"]),
                    float(row["Z"])
                ]
                self.loaded_points.append(p)
            print(f"Loaded {len(self.loaded_points)} existing points from {out_csv}")

        # Plot previously saved points
        if self.loaded_points:

            for (img_px, img_py, X, Y, Z) in self.loaded_points:
                # plot on input image
                self.ax1.plot(img_px, img_py, '*r')

                # plot on ortho
                self.ax2.plot(X, Y, '*r')

                # Add to the set of points
                self.points.append([img_px, img_py, X, Y, Z])

        self.fig.canvas.draw_idle()

    def start(self):
        img = self.img
        if img.shape[2] >= 3:
            rgb_img = img[:, :, :3]
        else:
            rgb_img = np.dstack([img[:, :, 0]] * 3)
        self.ax1.imshow(rgb_img.astype(np.float32) / np.max(rgb_img))
        self.ax1.set_title("Input image")

        # Orthoimage
        ortho = self.ortho
        if 'band' in ortho.dims:
            self.ax2.imshow(np.dstack([
                ortho.isel(band=0).data,
                ortho.isel(band=1).data,
                ortho.isel(band=2).data,
            ]),
            extent=(min(ortho.x), max(ortho.x), min(ortho.y), max(ortho.y))
            )
        else:
            self.ax2.imshow(
                ortho.data, cmap='Grays_r',
                extent=(min(ortho.x), max(ortho.x), min(ortho.y), max(ortho.y))
                )
        self.ax2.set_xlim(self.ortho_xlim)
        self.ax2.set_ylim(self.ortho_ylim)

        self.ax2.set_title("Orthoimage")

        plt.suptitle("Click match points: FIRST on IMAGE, THEN on ORTHO")
        plt.show()

    def onclick(self, event):
        if event.inaxes not in [self.ax1, self.ax2] or event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        # First click: input image
        if self.mode == "image" and event.inaxes == self.ax1:
            self.pending_img_pt = (x, y)
            print(f"[Image] px={x:.2f}, py={y:.2f}")
            self.mode = "ortho"
            return

        # Second click: orthoimage
        if self.mode == "ortho" and event.inaxes == self.ax2:
            X,Y = x, y
            print(f"[Ortho] X={x:.2f}, Y={y:.2f}")

            # DEM value
            Z = self.dem.sel(x=X, y=Y, method='nearest').data

            img_px, img_py = self.pending_img_pt
            self.points.append([img_px, img_py, X, Y, Z])
            print(f"--> Saved 3D: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

            # Plot points on image
            self.ax1.plot(img_px, img_py, '*r')

            # Plot points on orthoimage
            self.ax2.plot(X, Y, '*r')

            # Refresh canvas
            self.fig.canvas.draw_idle()

            # Reset for next point
            self.pending_img_pt = None
            self.mode = "image"

            # Reset for next point
            self.pending_img_pt = None
            self.mode = "image"

    def onscroll(self, event):
        ax = event.inaxes
        if ax not in [self.ax1, self.ax2]:
            return

        base_scale = 1.2 # zoom speed
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if event.button == 'up':
            scale = 1 / base_scale
        elif event.button == 'down':
            scale = base_scale
        else:
            return

        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx),
                     xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely),
                     ydata + new_height * rely])
        ax.figure.canvas.draw_idle()

    def onpress_pan(self, event):
        if event.inaxes in [self.ax1, self.ax2] and event.button in [2, 3]:
            self._pan_ax = event.inaxes
            self._pan_start = (event.xdata, event.ydata,
                               event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def onpan(self, event):
        if not hasattr(self, "_pan_ax") or event.inaxes != self._pan_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0, (xmin0, xmax0), (ymin0, ymax0) = self._pan_start
        dx = event.xdata - x0
        dy = event.ydata - y0
        ax = self._pan_ax
        ax.set_xlim(xmin0 - dx, xmax0 - dx)
        ax.set_ylim(ymin0 - dy, ymax0 - dy)
        ax.figure.canvas.draw_idle()

    def onrelease_pan(self, event):
        if hasattr(self, "_pan_ax"):
            del self._pan_ax

    def save(self):
        if len(self.points) < 1:
            print("No points selected, none saved.")
            return

        df = pd.DataFrame(self.points, columns=["img_px", "img_py", "X", "Y", "Z"])
        df.to_csv(self.out_csv, index=False)

        print(f"Saved {len(self.points)} total points → {self.out_csv}")


if __name__ == "__main__":

    ch = 22
    ch_string = f"ch{ch}" if ch >= 10 else f"ch0{ch}"
    data_path = '/Users/rdcrlrka/Research/Soo_locks'
    out_path = os.path.join(data_path, 'inputs', 'gcp') 

    image_file = os.path.join(data_path, 'inputs', 'original_images', f"{ch_string}_20251120140600.tiff")
    ortho_file = os.path.join(data_path, 'inputs', 'TLS_reflectance.tif') #'TLS_RGB.vrt')
    dem_file = os.path.join(data_path, 'inputs', 'TLS_DTM_cropped_filled.tif')
    cams_file = os.path.join(data_path, 'inputs', 'Camera_Locations.geojson')
    out_file = os.path.join(out_path, f"{ch_string}_gcp.csv")

    picker = PointPicker(
        img_path=image_file,
        ortho_path=ortho_file,
        dem_path=dem_file,
        cams_path=cams_file,
        out_csv=out_file
    )

    picker.start()
    picker.save()
