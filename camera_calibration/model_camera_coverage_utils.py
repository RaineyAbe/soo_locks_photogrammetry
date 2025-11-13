#! /usr/env/bin python

import os
import pandas as pd
import rioxarray as rxr
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely import get_coordinates
from shapely.ops import unary_union
import shapely
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio as rio
import matplotlib
import matplotlib.pyplot as plt
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


def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def create_new_footprint(
    camera_xyz,
    roll_deg=0,
    pitch_deg=0,
    yaw_deg=15,
    fov_h_deg=120,
    fov_v_deg=65,
    ground_z=-8,
    image_width=4512,
    image_height=2512,
    trusses=None,
    model_space=None
    ):
    X, Y, Z = camera_xyz
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    # --- Camera intrinsics ---
    fx = image_width / (2 * np.tan(np.deg2rad(fov_h_deg / 2)))
    fy = image_height / (2 * np.tan(np.deg2rad(fov_v_deg / 2)))
    cx, cy = image_width / 2, image_height / 2
    K = np.array([
        [fx, 0, cx], 
        [0, fy, cy], 
        [0, 0, 1]
        ], dtype=np.float64)

    # --- Camera rotation/translation ---
    R_cw = rpy_to_matrix(roll, pitch, yaw)

    # --- Project image corners to ground plane ---
    image_corners = np.array([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1]
    ], dtype=np.float32)

    world_footprint = []
    for u, v in image_corners:
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_cam = np.array([x, y, 1.0])
        ray_world = R_cw @ ray_cam
        s = (ground_z - Z) / ray_world[2]
        intersection = np.array([X, Y, Z]) + s * ray_world
        world_footprint.append(intersection)

    world_footprint = np.array(world_footprint)
    footprint_poly = Polygon(world_footprint[:, :2])

    model_space_geom = model_space['geometry'].values[0].buffer(-0.05)

    # Iterate over trusses (if cameras aren't nadir)
    if (trusses is not None) & (roll!=0):
        for _,truss_row in trusses.iterrows():
            footprint_split_half = None
            extended_line = None
            truss_coords = np.array(truss_row['geometry'].exterior.coords, dtype=np.float32)

            # get the bottom
            ibottom = np.argwhere(truss_coords[:,2]==min(truss_coords[:,2])).reshape(1,-1)
            truss_bottom = np.unique(truss_coords[ibottom], axis=1)[0]

            # extend truss bottom plane from camera to ground height
            extended_pts = []
            for p in truss_bottom:
                direction = p - camera_xyz
                t = (ground_z - Z) / direction[2]
                p_ground = camera_xyz + t * direction
                extended_pts.append(p_ground)
            extended_pts = np.array(extended_pts)

            # create linestring in the XY plane
            extended_line = LineString(extended_pts[:,:2])

            # Scale to make sure it crosses the image footprint
            extended_line = shapely.affinity.scale(extended_line, xfact = 5, yfact = 5)

            # now, use it to split the footprint
            footprint_split = shapely.ops.split(footprint_poly, extended_line)
            
            # continue if the footprint wasn't split (truss line didn't intersect footprint)
            if len(footprint_split.geoms) < 2:
                continue
            
            # identify the "north" vs. "south" halves of the split
            footprint_split_centroids_y = [x.centroid.coords.xy[1] for x in footprint_split.geoms]
            if footprint_split_centroids_y[0] > footprint_split_centroids_y[1]:
                footprint_split_N = footprint_split.geoms[0]
                footprint_split_S = footprint_split.geoms[1]
            else:
                footprint_split_N = footprint_split.geoms[1]
                footprint_split_S = footprint_split.geoms[0]

            # select the split half based on viewing angle
            if roll > 0: # positive viewing angle
                footprint_split_half = footprint_split_S
            elif roll < 0: # negative viewing angle
                footprint_split_half = footprint_split_N

            footprint_poly = footprint_split_half

    # Crop to model space
    footprint_poly = footprint_poly.intersection(model_space_geom)

    return footprint_poly


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
                        bounds_color='#8da0cb', overlap_color='#8da0cb', missing_color='#fc8d62', new_cam_color="#1b9e77"):
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
              markerfacecolor='None', markeredgecolor=new_cam_color, markeredgewidth=2, markersize=12, label='NEW cameras')
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
                       bounds_color='#8da0cb', new_cam_color='#1b9e77'):
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
            axis.plot(row['Y'], row['Z'], '*', markerfacecolor='None', markeredgecolor=new_cam_color, markeredgewidth=2, markersize=10)
    
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


def save_specs_los(cameras_xyz, rolls, pitches, yaws, out_file, fov_h=120, fov_v=65):
    # Estimate FOV
    specs_list = []
    for i in range(len(cameras_xyz)):
        # compile in dataframe
        df = pd.DataFrame({
            'X': cameras_xyz[i,0],
            'Y': cameras_xyz[i,1],
            'Z': cameras_xyz[i,2],
            'FOV_vertical': fov_v,
            'FOV_horizontal': fov_h,
            'roll': rolls[i],
            'pitch': pitches[i],
            'yaw': yaws[i]
        }, index=[i])
        specs_list += [df]

    specs_df = pd.concat(specs_list)
    specs_df = specs_df.round(2)

    # save
    specs_df.to_csv(out_file, header=True, index=False)
    print("New camera specs saved to file:", out_file)
    return


def calculate_new_coverage(
        new_coords: np.array = None, 
        new_rolls: np.array = None, 
        new_pitches: np.array = None, 
        new_yaws: np.array = None, 
        cams: pd.DataFrame = None, 
        bounds: gpd.GeoDataFrame = None, 
        fov_h: float = 120, 
        fov_v: float = 65,
        trusses: gpd.GeoDataFrame = None,
        model_space: gpd.GeoDataFrame = None, 
        out_folder: str = None,
        label: str = None
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
        print('No new rolls provided, setting all to 0.')
    if new_pitches is None:
        new_pitches = np.zeros(n_new)
        print('No new pitches provided, setting all to 0.')
    if new_yaws is None:
        new_yaws = 15*np.ones(n_new)
        print('No new yaws provided, setting all to 15.')
        
    # Full arrays: existing cameras assumed nadir (0 roll/pitch) + yaw=15 deg
    rolls_full = np.concatenate([np.zeros(n_existing), new_rolls])
    pitches_full = np.concatenate([np.zeros(n_existing), new_pitches])
    yaws_full = np.concatenate([15*np.ones(n_existing), new_yaws])
    
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
            trusses=trusses
        )
        bounds_new_list.append(gpd.GeoDataFrame({
            'geometry': [footprint],
            'channel': [cams_new_full['channel'].iloc[i]]
        }))

    bounds_new = pd.concat(bounds_new_list[n_existing:]).reset_index(drop=True)
    bounds_new['geometry'] = bounds_new['geometry']
    bounds_new_full = pd.concat([bounds, bounds_new]).reset_index(drop=True)
    bounds_new_full['geometry'] = bounds_new_full['geometry']
    
    # --- Recalculate image overlap ---
    overlap_new = calculate_image_overlap(bounds_new_full) 

    # --- Identify model space with no coverage ---
    no_coverage_new_full = calculate_no_coverage(model_space, bounds_new_full) 
    print('No coverage area = ', np.round(no_coverage_new_full.area,1), 'm^2') 

    # --- Calculate relative distortion ---
    distortion_map = create_distortion_map(
        cams_gdf = cams_new_full,
        footprints_gdf = bounds_new_full,
        yaw_series = 15 * np.ones(len(cams_new_full)),
        pitch_series = np.zeros(len(cams_new_full)),
        roll_series = np.zeros(len(cams_new_full)),
        fov_h_deg = fov_h,
        fov_v_deg = fov_v,
        model_space = model_space
    )
    # add the minimum value
    distortion_map += 5.297444280721465

    # --- Plot results ---
    gs = matplotlib.gridspec.GridSpec(2,2, height_ratios=[4,1])
    fig = plt.figure(figsize=(10,14))
    ax = [
        fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,:])
        ]
    # plot model coverage
    plot_model_coverage(
        axis = ax[0],
        model_space = model_space,
        bounds = bounds_new_full,
        overlap = overlap_new,
        cam_positions = cams_new_full,
        no_coverage = no_coverage_new_full
        )
    ax[0].legend(loc='lower left')
    # plot relative distortion
    distortion_map.plot(ax=ax[1], cmap='Reds', vmin=0, vmax=1.5, cbar_kwargs={'shrink': 0.5})
    # print distortion stats
    ax[1].text(-28, 75, f"Mean = {np.round(distortion_map.mean().data, 2)}")
    ax[1].text(-28, 73, f"Median = {np.round(distortion_map.median().data, 2)}")
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].set_xlabel('X [meters]')
    ax[1].set_ylabel('')
    ax[1].set_title('')
    # plot vertical view
    plot_vertical_view(
        axis = ax[2], 
        trusses_gdf = trusses,
        cams = cams,
        cams_new = cams_new,
        bounds = bounds,
        bounds_new = bounds_new
        )

    fig.suptitle(label)
    fig.tight_layout()
    plt.show()

    # --- Save results ---
    label_file = label.replace(' ','_').lower()
    # figure
    fig_file = os.path.join(out_folder, f"{label_file}.png")
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    print('Figure saved to file:', fig_file)

    # Save new specs
    out_file = os.path.join(out_folder, f"{label_file}_specs.csv")
    save_specs_los(
        cameras_xyz = new_coords,
        rolls = new_rolls,
        pitches = new_pitches,
        yaws = new_yaws,
        out_file = out_file
    )
    
    return
