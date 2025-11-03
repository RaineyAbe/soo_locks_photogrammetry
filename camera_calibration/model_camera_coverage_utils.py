#! /usr/env/bin python

import os
import pandas as pd
import rioxarray as rxr
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import get_coordinates
from shapely.ops import unary_union
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio as rio
import matplotlib
import cv2

def calculate_image_footprint(raster_file):
    # create a mask of data coverage
    raster = rxr.open_rasterio(raster_file).isel(band=0)
    crs, transform = raster.rio.crs, raster.rio.transform()
    if 'DSM' in raster_file:
        raster = xr.where(raster!=-9999, raster, np.nan)
    else:
        raster = xr.where(raster > 0, raster, np.nan)
    mask = raster.notnull()

    # vectorize the mask
    shape_gen = (
        (shape(s), v) 
        for s, v in 
        rio.features.shapes(mask.values.astype(np.int8), transform=transform)
        ) 
    gdf = gpd.GeoDataFrame(dict(zip(["geometry", "mask"], zip(*shape_gen))), crs=crs)

    # use just the exterior of all polygons
    exteriors = [x.exterior for x in gdf['geometry']]
    exterior_polys = [Polygon(get_coordinates(x)) for x in exteriors]
    gdf['geometry'] = exterior_polys

    # add the camera "channel"
    gdf['channel'] = os.path.basename(raster_file).split('_')[0]
    
    # use the largest mask polygon
    gdf['area'] = [x.area for x in gdf.geometry]
    gdf = gdf.sort_values(by='area', ascending=False).reset_index(drop=True)
    gdf = gdf.iloc[0:3]
    gdf = gdf.loc[gdf['mask']==1].reset_index(drop=True)   

    # make sure there's only one remaining
    gdf = gpd.GeoDataFrame(gdf.iloc[0]).transpose()

    # buffer the polygon slightly to remove sharp divots
    gdf['geometry'] = [x.buffer(0.1, join_style=1) for x in gdf['geometry']]

    # recalculate area
    gdf['area'] = [x.area for x in gdf['geometry']]

    gdf = gdf.set_geometry('geometry', crs=crs)

    return gdf


def calculate_image_overlap(bounds_gdf, buffer=0.01):
    # Buffer to avoid slivers
    bounds_gdf['geometry'] = bounds_gdf['geometry'].buffer(buffer)
    
    overlaps = []
    for i in range(len(bounds_gdf)):
        poly1 = bounds_gdf.geometry.iloc[i]
        channel1 = bounds_gdf.channel.iloc[i]
        for j in range(i+1, len(bounds_gdf)):
            poly2 = bounds_gdf.geometry.iloc[j]
            channel2 = bounds_gdf.channel.iloc[j]
            intersection = poly1.intersection(poly2)
            if intersection.is_empty:
                continue
            overlaps.append({
                'channel_a': channel1,
                'channel_b': channel2,
                'overlap_area': intersection.area,
                'geometry': intersection
            })
    if overlaps:
        overlap_gdf = gpd.GeoDataFrame(overlaps, geometry='geometry', crs=bounds_gdf.crs)
    else:
        overlap_gdf = gpd.GeoDataFrame(columns=['channel_a','channel_b','overlap_area','geometry'], crs=bounds_gdf.crs)
    
    return overlap_gdf


def get_coords_between_cams(ch1, ch2, cams):
    cam1 = cams.loc[cams['channel']==ch1]
    cam2 = cams.loc[cams['channel']==ch2]
    xcoord = np.nanmean([cam1['X'].values[0], cam2['X'].values[0]])
    ycoord = np.nanmean([cam1['Y'].values[0], cam2['Y'].values[0]])
    return (xcoord, ycoord)


def calculate_no_coverage(model_space_gdf, bounds_gdf, buffer=0.01):
    # Buffer footprints slightly to close slivers
    buffered_bounds = bounds_gdf.copy()
    buffered_bounds['geometry'] = buffered_bounds.geometry.buffer(buffer)
    
    # Union of all footprints
    total_coverage = unary_union(buffered_bounds.geometry)
    
    # Union of model space polygons
    model_union = unary_union(model_space_gdf.geometry)
    
    # Compute uncovered area
    no_coverage = model_union.difference(total_coverage)
    
    return no_coverage


def create_new_footprint(
        camera_xyz,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=-15,
        fov_h_deg=120,
        fov_v_deg=65,
        ground_z=-8,
        image_width=4512,
        image_height=2512,
        trusses_gdf=None,
        model_space=None
):
    # Construct the camera rotation amtrix
    X, Y, Z = camera_xyz
    fx = image_width / (2 * np.tan(np.deg2rad(fov_h_deg / 2)))
    fy = image_height / (2 * np.tan(np.deg2rad(fov_v_deg / 2)))
    cx, cy = image_width / 2, image_height / 2
    roll, pitch, yaw = map(np.deg2rad, [roll_deg, pitch_deg, yaw_deg])
    R = cv2.Rodrigues(np.array([roll, pitch, yaw]))[0]

    # Image corners (pixel coords)
    image_corners = np.array([
        [0, 0],                                 # lower left
        [image_width - 1, 0],                   # lower right
        [image_width - 1, image_height - 1],    # upper right
        [0, image_height - 1],                  # upper left
    ], dtype=np.float32)

    # Project each corner to ground plane
    world_footprint = []
    for u, v in image_corners:
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_cam = np.array([x, y, 1.0])
        ray_world = R.T @ ray_cam
        s = (ground_z - Z) / ray_world[2]
        intersection = np.array([X, Y, Z]) + s * ray_world
        world_footprint.append(intersection)
    world_footprint = np.array(world_footprint)

    # Check for truss intersections (YZ plane)
    if trusses_gdf is not None:
        # Compute camera forward direction (horizontal)
        forward_world = R.T @ np.array([0, 0, 1])
        forward_xy = forward_world[:2]
        norm_xy = np.linalg.norm(forward_xy)
        # normalize, avoiding division by zero
        if norm_xy < 1e-9:
            forward_xy[:] = np.array([0, 1])
        else:
            forward_xy /= norm_xy

        # create YZ polygon that includes camera position 
        world_footprint_yz = Polygon([ 
            (Y, Z), # camera position 
            (min(world_footprint[:,1]), ground_z), # front edge 
            (max(world_footprint[:,1]), ground_z), # back edge 
            (Y, Z) # close the polygon 
            ])

        dy_N, dy_S = 0, 0
        for _, truss_row in trusses_gdf.iterrows():
            truss = truss_row.geometry
            truss_y = [c[1] for c in truss.exterior.coords]
            truss_z = [c[2] for c in truss.exterior.coords]
            truss_yz = Polygon(np.column_stack((truss_y, truss_z)))

            if not world_footprint_yz.intersects(truss_yz):
                continue

            truss_bottom = np.min(truss_z)
            if Y < np.min(truss_y):  # looking toward +Y
                theta = np.arctan((Z - truss_bottom) / (np.min(truss_y) - Y))
                dy = -(Z - truss_bottom) / np.tan(theta)
                dy_N = min(dy_N, dy)
            elif Y > np.max(truss_y):  # looking toward -Y
                theta = np.arctan((Z - truss_bottom) / (Y - np.max(truss_y)))
                dy = (Z - truss_bottom) / np.tan(theta)
                dy_S = max(dy_S, dy)
        
        # Add the y adjustments
        # sort by northness
        sorted_indices = world_footprint[:, 1].argsort()
        world_footprint = world_footprint[sorted_indices]
        # add dy_S to the southernmost
        # world_footprint[-2:, 1] += dy_S
        # subtract dy_N from the northernmost
        # world_footprint[0:2, 1] -= dy_N
        print(dy_S, dy_N)

        # undo the sort to avoid invalid geometries
        unsorted_indices = np.argsort(sorted_indices)
        world_footprint = world_footprint[unsorted_indices]

    # Construct polygon
    footprint_poly = Polygon(world_footprint[:, :2])

    if model_space is not None:
        footprint_poly = footprint_poly.intersection(model_space.geometry.values[0])

    return footprint_poly


def calculate_specs_from_new_coords(
        new_coords: np.array = None, 
        new_rolls: np.array = None, 
        new_pitches: np.array = None, 
        new_yaws: np.array = None, 
        cams: pd.DataFrame = None, 
        bounds: gpd.GeoDataFrame = None, 
        fov_h: float = 120, 
        fov_v: float = 65,
        trusses_gdf: gpd.GeoDataFrame = None,
        model_space: gpd.GeoDataFrame = None, 
        ):
    
    # --- Create new cameras dataframe ---
    new_channels = np.arange(len(new_coords)) + 1 + len(cams)
    new_channels_string = [f"ch{x}" if x >=10 else f"ch0{x}" for x in new_channels]

    cams_new = pd.DataFrame({
        'img_name': ['None']*len(new_coords),
        'X': new_coords[:,0],
        'Y': new_coords[:,1],
        'Z': new_coords[:,2],
        'X_std': [0.1]*len(new_coords),
        'Y_std': [0.1]*len(new_coords),
        'Z_std': [0.1]*len(new_coords),
        'channel': new_channels_string
    })
    
    # Combine original + new cameras
    cams_new_full = pd.concat([cams, cams_new]).reset_index(drop=True)
    n_total = len(cams_new_full)
    n_existing = len(cams)
    n_new = len(cams_new)
    
    # --- Ensure rotation arrays are correctly sized ---
    if new_rolls is None:
        new_rolls = np.zeros(n_new)
    if new_pitches is None:
        new_pitches = np.zeros(n_new)
    if new_yaws is None:
        new_yaws = -15*np.ones(n_new)
        
    # Full arrays: existing cameras assumed nadir (0 roll/pitch) + yaw=-15 deg
    rolls_full = np.concatenate([np.zeros(n_existing), new_rolls])
    pitches_full = np.concatenate([np.zeros(n_existing), new_pitches])
    yaws_full = np.concatenate([-15*np.ones(n_existing), new_yaws])
    
    # --- Calculate footprints for all cameras ---
    bounds_new_list = []
    for i in range(n_total):
        cam_pos = cams_new_full[['X','Y','Z']].iloc[i].values
        roll, pitch, yaw = rolls_full[i], pitches_full[i], yaws_full[i]
        footprint = create_new_footprint(
            camera_xyz=cam_pos,
            model_space=model_space,
            roll_deg=roll,
            pitch_deg=pitch,
            yaw_deg=yaw,
            fov_h_deg=fov_h,
            fov_v_deg=fov_v,
            trusses_gdf=trusses_gdf
        )
        bounds_new_list.append(gpd.GeoDataFrame({
            'geometry': [footprint],
            'channel': [cams_new_full['channel'].iloc[i]]
        }))

    bounds_new_gdf = pd.concat(bounds_new_list[n_existing:]).reset_index(drop=True)
    bounds_new_gdf['geometry'] = bounds_new_gdf['geometry']
    bounds_new_full_gdf = pd.concat([bounds, bounds_new_gdf]).reset_index(drop=True)
    bounds_new_full_gdf['geometry'] = bounds_new_full_gdf['geometry']
    
    # recalculate image overlap 
    overlap_new_gdf = calculate_image_overlap(bounds_new_full_gdf) 

    # identify model space with no coverage 
    no_coverage_new_full = calculate_no_coverage(model_space, bounds_new_full_gdf) 
    print('No coverage area = ', np.round(no_coverage_new_full.area,1), 'm^2') 
    
    return cams_new, cams_new_full, bounds_new_gdf, bounds_new_full_gdf, overlap_new_gdf, no_coverage_new_full


def create_distortion_map(
    cams_gdf,
    footprints_gdf,
    yaw_series,
    pitch_series,
    roll_series,
    fov_h_deg=120,
    fov_v_deg=65,
    dx=0.05,
    dy=0.05,
    model_floor_z=-8,
    model_space=None
):
    """
    Computes a raster of relative distortion based on angular distance from each camera's optical center.
    Overlapping images are merged using the minimum distortion at each pixel.
    """
    # Raster bounds
    x_min, y_min, x_max, y_max = footprints_gdf.total_bounds
    xs = np.arange(x_min, x_max + dx, dx)
    ys = np.arange(y_max, y_min - dy, -dy)  # descending for raster origin
    shape = (len(ys), len(xs))
    
    # Initialize raster
    out_raster = np.full(shape, np.nan, dtype=float)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    
    # Lowest distortion at half the FOV angles
    half_h = np.deg2rad(fov_h_deg/2)
    half_v = np.deg2rad(fov_v_deg/2)
    
    # Loop over cameras
    for i, row in footprints_gdf.iterrows():
        cam = cams_gdf.iloc[i]
        footprint = row.geometry
        roll, pitch, yaw = map(np.deg2rad, [roll_series[i], pitch_series[i], yaw_series[i]])
        Rcw = cv2.Rodrigues(np.array([roll, pitch, yaw]))[0]
        Rwc = Rcw.T
        cam_pos = cam[['X','Y','Z']].values.astype(float)
        
        # Rasterize footprint
        transform = rio.transform.from_origin(x_min, y_max, dx, dy)
        mask = rio.features.rasterize(
            [(footprint,1)], 
            out_shape=shape, 
            transform=transform,
            fill=0, 
            all_touched=True
            ).astype(bool)
        if not mask.any():
            print('Footprint not valid, skipping.')
            continue
        
        # World coordinates of raster pixels in footprint
        px = xx[mask]
        py = yy[mask]
        pz = np.full_like(px, model_floor_z)
        points_xyz = np.vstack([px, py, pz]).T
        
        # Vector from camera to points
        v = points_xyz - cam_pos
        v_cam = (Rwc @ v.T).T  # world -> camera
        
        # Keep only points in front of camera
        visible = v_cam[:,2] < 0
        if np.sum(visible) == 0:
            continue
        v_cam_vis = v_cam[visible]
        mask_indices = np.flatnonzero(mask)[visible]
        
        # Calculate distance from optical center
        alpha = np.arctan2(v_cam_vis[:,0], v_cam_vis[:,2])
        beta  = np.arctan2(v_cam_vis[:,1], v_cam_vis[:,2])
        norm_radius = np.sqrt((alpha/half_h)**2 + (beta/half_v)**2)

        # Convert to distortion: low at center, high at edges
        distortion = 1 - norm_radius
        
        # Merge into output raster (min distortion)
        current_vals = out_raster.flat[mask_indices]
        current_vals = np.where(np.isnan(current_vals), distortion, np.minimum(current_vals, distortion))
        out_raster.flat[mask_indices] = current_vals
    
    # Construct xarray DataArray
    distortion_map = xr.DataArray(
        out_raster, 
        coords={'y': ys, 'x': xs}, 
        dims=('y','x'), 
        name='distortion'
        )
    
    # Clip to model space
    distortion_map.rio.write_crs("EPSG:32619", inplace=True)
    distortion_map = distortion_map.rio.clip(model_space['geometry'])
    
    return distortion_map

    
def plot_model_coverage(axis, model_space, bounds, overlap, cam_positions, no_coverage,
                        bounds_color='#7570b3', overlap_color='#7570b3', missing_color='#d95f02', new_cam_color='#e7298a'):
    # model space
    model_space.plot(ax=axis, edgecolor='k', facecolor='None', linewidth=2)
    # image footprints
    bounds.plot(ax=axis, edgecolor=bounds_color, linewidth=2, linestyle='--', facecolor='None', legend=False)
    # image overlap
    overlap.plot(ax=axis, color=overlap_color, alpha=0.5, legend=False)
    # missing model coverage
    no_coverage_gdf = gpd.GeoDataFrame(geometry=[no_coverage], crs="EPSG:32619")
    no_coverage_gdf.plot(ax=axis, edgecolor='None', facecolor=missing_color, alpha=0.5, legend=False)
    # current and NEW camera positions
    cam_positions_current = cam_positions.iloc[0:16]
    axis.plot(cam_positions_current['X'].values, cam_positions_current['Y'].values, '*k', markersize=8, label='Cameras')
    cam_positions_new = cam_positions.iloc[16:]
    axis.plot(cam_positions_new['X'].values, cam_positions_new['Y'].values, '*', 
              markerfacecolor='None', markeredgecolor=new_cam_color, linewidth=1.5, markersize=12, label='NEW cameras')
    # dummy points for legend
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    axis.plot(1e3, 1e3, '-k', linewidth=2,label='Model space')
    axis.plot(1e3, 1e3, '--', linewidth=2, color=bounds_color, label='Image footprint')
    axis.plot(1e3, 1e3, 's', markerfacecolor=overlap_color, alpha=0.5, markeredgecolor='None', markersize=10, label='Image overlap')
    axis.plot(1e3, 1e3, 's', markerfacecolor=missing_color, alpha=0.5, markeredgecolor='None', markersize=10, label='No coverage')
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_xlabel('X [meters]')
    axis.set_ylabel('Y [meters]')
    return


def plot_vertical_view(axis, trusses_gdf, cams = None, cams_new = None, bounds = None, bounds_new = None,
                       bounds_color='#7570b3', new_cam_color='#e7298a'):
    # trusses
    for _,row in trusses_gdf.iterrows():
        truss_y = np.array([coord[1] for coord in row['geometry'].exterior.coords])
        truss_ymean = np.nanmean(truss_y)
        truss_z = np.array([coord[2] for coord in row['geometry'].exterior.coords])
        truss_zmin = np.nanmin(truss_z)
        axis.plot([truss_ymean, truss_ymean], [truss_zmin, -3], '-k')

    # current camera footprints
    def plot_cam_footprints(b_gdf, c_gdf):
        for i,row in b_gdf.iterrows():
            bound = row['geometry']
            if type(bound)==MultiPolygon:
                bound_y = np.array([])
                for geom in bound.geoms:
                    _, geom_y = geom.exterior.coords.xy
                    bound_y = np.append(bound_y, geom_y)
            else:
                _, bound_y = bound.exterior.coords.xy
            bound_ymin, bound_ymax = np.min(bound_y), np.max(bound_y)
            cam_y, cam_z = c_gdf.iloc[i][['Y', 'Z']].values
            bound_poly = matplotlib.patches.Polygon(
                np.array([[bound_ymin, -8],[bound_ymax, -8], [cam_y, cam_z], [bound_ymin, -8]]),
                facecolor=bounds_color, alpha=0.3, edgecolor='None'
            )
            axis.add_patch(bound_poly)
    plot_cam_footprints(bounds, cams)
    if cams_new is not None:
        plot_cam_footprints(bounds_new, cams_new)

    # current camera positions
    for i,row in cams.iterrows():
        axis.plot(row['Y'], row['Z'], '*k', markersize=10)

    # new camera positions
    if cams_new is not None:
        for i,row in cams_new.iterrows():
            axis.plot(row['Y'], row['Z'], '*', markerfacecolor='None', markeredgecolor=new_cam_color, linewidth=1.5, markersize=10)
    
    # ceiling
    axis.plot([0,70], [-3,-3], '-k')
    # model ground-ish
    axis.plot([0,70], [-8,-8], '-k')
    # add some text labels
    axis.text(70,-7.8, '~Model', horizontalalignment='right', fontsize=12, color='k')
    axis.text(70,-5.4, 'Trusses', horizontalalignment='right', fontsize=12)
    axis.text(70,-3.5, 'Ceiling', horizontalalignment='right', fontsize=12)
    axis.set_xlabel('Y [meters]')
    axis.set_ylabel('Z [meters]')

    return


def save_specs_los(bounds_new, cams_new, cams, out_file, fov_h=120, fov_v=65):
    # Estimate FOV
    rotation = -13
    yaw = 360 + rotation
    cam_height = float(cams['Z'].mean()) + 8
    specs_list = []
    for i,_ in bounds_new.iterrows():
        cam = cams_new.iloc[i]
        # compile in dataframe
        df = pd.DataFrame({
            'new_cam_number': i+1,
            'X': cam['X'],
            'Y': cam['Y'],
            'Z': cam_height,
            'FOV_vertical': fov_v,
            'FOV_horizontal': fov_h,
            'roll': 0,
            'pitch': 0,
            'yaw': yaw
        }, index=[i])
        specs_list += [df]

    specs_df = pd.concat(specs_list)
    specs_df = specs_df.round(2)

    # save
    specs_df.to_csv(out_file, header=True, index=False)
    print("New camera specs saved to file:", out_file)
    return

