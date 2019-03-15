from fastai.vision import *
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import gdal
from osgeo import gdal_array
from osgeo import osr
import ast
from glob import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import re
import subprocess
from functools import partial
import shutil

def parallelize(fn, it, do_multiprocessing=False, n_workers=defaults.cpus):
    executor = ProcessPoolExecutor if do_multiprocessing else ThreadPoolExecutor
    with executor(n_workers) as e:
        result = e.map(fn, it)
    return result

def get_hw(mos_pth):
    ds = gdal.Open(mos_pth)
    band = ds.GetRasterBand(1)
    return band.XSize, band.YSize

def resize(raw_mos_pth, out_pth, output_pixel_size=0.4):
    ds = gdal.Open(raw_mos_pth)
    tfms_params = ds.GetGeoTransform()
    pixel_size = (tfms_params[1], abs(tfms_params[5]))
    raster_size = (ds.RasterXSize, ds.RasterYSize)    
    
    outsize0 = int(pixel_size[0] * raster_size[0] * (1/output_pixel_size))
    outsize1 = int(pixel_size[1] * raster_size[1] * (1/output_pixel_size))
    
    bashCommand = f'gdalwarp -ts {outsize0} {outsize1} {raw_mos_pth} {out_pth}'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE) 
    output, error = process.communicate()

def generate_tile_df(mos_pth, tile_size, step_ratio):
    out_file_prefix = f'tile_'
    
    mos_xsize, mos_ysize = get_hw(mos_pth)
    
    step = int(step_ratio * tile_size)
    cnt = 0
    
    df = pd.DataFrame(columns=['mos_tile_name', 'x', 'y', 'tile_size'])
    
    for x in range(0, mos_xsize-tile_size, step):
        for y in range(0, mos_ysize-tile_size, step):
            cnt = cnt + 1
            
            mos_file_name = f'{out_file_prefix}_{cnt}_{x}_{y}.tif'
            
            tile_data = {
                'mos_tile_name': mos_file_name,
                'x': x,
                'y': y,
                'tile_size': tile_size,
                'step_ratio': step_ratio
            }
            df = df.append(pd.Series(tile_data), ignore_index=True)
    
    return df

def generate_tile(x, y, tile_size, img_pth, file_name, out_folder):
    out_pth = f'{out_folder}/{file_name}'
    com_string = f'gdal_translate -srcwin {x}, {y}, {tile_size}, {tile_size}, {img_pth} {out_pth}'
    os.system(com_string)

def generate_tile_from_row(row, mos_pth, out_folder):
    generate_tile(row.x, row.y, row.tile_size, mos_pth, row.mos_tile_name, out_folder)

def generate_tiles(tile_df, tile_folder, mos_pth):
    fn = partial(generate_tile_from_row, mos_pth=mos_pth, out_folder=tile_folder)
    with ThreadPoolExecutor(max_workers=8) as m:
        result = m.map(fn, tile_df.itertuples())

def get_nodata_val(mos_pth):
    no_data = None
    ds = gdal.Open(mos_pth)
    band = ds.GetRasterBand(1)
    mos_xsize = band.XSize
    mos_ysize = band.YSize
    corner_vals = []
    for x in [0, mos_xsize - 1]:
        for y in [0, mos_ysize - 1]:
            result = os.popen(f'gdallocationinfo -valonly {mos_pth} {x} {y}').read()
            corner_vals.append(' '.join(result.split())) 
    corner_vals = pd.Series(corner_vals).value_counts()
    if corner_vals.iloc[0] > 2:
        no_data = list(map(int, corner_vals.index[0].split()[:-1]))
    return no_data

def is_no_data_tile(tile_name, no_data_val, tile_folder):
    img = cv2.cvtColor(cv2.imread(f'{tile_folder}/{tile_name}'), cv2.COLOR_BGR2RGB)
    return np.all(img == no_data_val)

def remove_small_cc(mask_img, min_area):
    total_comp, label_img, stats, centroids = cv2.connectedComponentsWithStats(mask_img.astype(np.uint8))
    areas = stats[:, -1]
    comp_to_use = np.where(areas > min_area)[0]
    print('Total Connected Components:', total_comp)
    print(f'{total_comp - len(comp_to_use)} components removed')
    final_mask = label_img.copy()
    for label in range(total_comp):
        if label not in comp_to_use:
            # Label `0` is always background
            final_mask[final_mask == label] = 0
    final_mask[final_mask != 0] = 1
    return final_mask

def array2raster(newRasterfn, src_ds_pth, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    dataset = gdal.Open(src_ds_pth)
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    elif dtype == "Int32":
        GDT_dtype = gdal.GDT_Int32
    elif dtype == 'Bool':
        GDT_dtype = gdal.GDT_Byte
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

'''
Currently only works with step_ratio=0.5
'''
class MosaicSegmentationModel:
    
    def __init__(self, model_pth = 'cdc-0.4-256.pkl', temp_folder='tmp', terrain_index=1, bs=8):
        self.model_pth = model_pth
        self.terrain_index = terrain_index
        self.bs = bs
        self.path = temp_folder
        self.tile_folder = 'mos_tiles'
    
    def predict_mosaic(self, mos_pth, pixel_size, tile_size, step_ratio=0.5):
        print('Preparing data for model')
        self.data = self.prepare_data(mos_pth, pixel_size, tile_size, step_ratio)
        print('Load model and get predictions')
        self.learner = load_learner('.', self.model_pth, test=self.data, bs=self.bs)
        probs, target = self.learner.get_preds(ds_type=DatasetType.Test)
        print('Making probability map for resized mosaic')
        prob_map = self.make_prob_map(probs, tile_size, step_ratio)
        return prob_map
        
    
    def prepare_data(self, mos_pth, pixel_size, tile_size, step_ratio):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(f'{self.path}/{self.tile_folder}')
        
        print(f'Resizing mosaic to {pixel_size} pixel_size')
        resized_mos_pth = f'{self.path}/resized_mos.tiff'
        resize(mos_pth, resized_mos_pth, output_pixel_size=pixel_size)
        print('Generating Tiles')
        self.tile_df = generate_tile_df(resized_mos_pth, tile_size=tile_size, step_ratio=step_ratio)
        generate_tiles(self.tile_df, f'{self.path}/{self.tile_folder}', resized_mos_pth)
        
        print('Removing No data tiles')
        no_data = get_nodata_val(mos_pth)
        fn = partial(is_no_data_tile, no_data_val=no_data, tile_folder=f'{self.path}/{self.tile_folder}')
        is_nodata = list(parallelize(fn, self.tile_df.mos_tile_name, do_multiprocessing=True))
        is_nodata = np.array(is_nodata)
        self.tile_df = self.tile_df[~is_nodata]
        print(f'{is_nodata.sum()} tiles removed')
        self.tile_df.reset_index(drop=True, inplace=True)
        
        return SegmentationItemList.from_df(self.tile_df, path=self.path, folder=Path('mos_tiles'), cols=['mos_tile_name'])
    
    def make_prob_map(self, probs, tile_size, step_ratio):
        probs = probs.permute(0, 2, 3, 1)
        terrain_probs = probs[:, :, :, self.terrain_index]
        terrain_probs = np.array(terrain_probs)
        mos_ysize, mos_xsize = get_hw(f'{self.path}/resized_mos.tiff')
        terrain_map = -1 * np.ones((mos_xsize, mos_ysize, 4))
        
        for index, row in self.tile_df.iterrows():
            prob_map = terrain_probs[index]
            subtile_size = tile_size//2
            for x in [0, subtile_size]:
                for y in [0, subtile_size]:
                    subtile = prob_map[y:y+subtile_size, x:x+subtile_size]
                    loc = (row.x + x, row.y + y)
                    for channel in range(0, 4):
                        if np.all(terrain_map[loc[1]:loc[1] + subtile_size, loc[0]:loc[0] + subtile_size, channel] == -1):
                            terrain_map[loc[1]:loc[1] + subtile_size, loc[0]:loc[0] + subtile_size, channel] = subtile
                            break
        
        terrain_map[terrain_map == -1] = np.nan
        terrain_map_min = np.nanmin(terrain_map, axis=2)
        terrain_map_min = np.nan_to_num(terrain_map_min)
        return terrain_map_min

class TerrainDetection:
    
    def __init__(self, seg_model, pixel_size, tile_size, step_ratio=0.5):
        self.seg_model = seg_model
        self.pixel_size = pixel_size
        self.tile_size = tile_size
        self.step_ratio = step_ratio
    
    def make_terrain_shape_file(self, mos_pth, out_folder, threshold, min_area=500):
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        mos_name = Path(mos_pth).stem
        self.mask_pth = f'tmp/{mos_name}_mask.tif'
        self.small_mask_pth = f'tmp/{mos_name}_small_mask.tif'
        self.shape_pth = f'{out_folder}/{mos_name}_shape.shp'
        self.small_shape_pth = f'{out_folder}/{mos_name}_small_shape.shp'
        self.prob_map = self.seg_model.predict_mosaic(mos_pth, self.pixel_size, 
                                                 self.tile_size, self.step_ratio)
        self.terrain_mask = self.prob_map > threshold
        self.terrain_mask = remove_small_cc(self.terrain_mask, min_area)
        
        
        print('Saving mask as geo-tiff')
        array2raster(self.small_mask_pth, 'tmp/resized_mos.tiff', self.terrain_mask, 'Bool')
        outx, outy = get_hw(mos_pth)
        print('Resizing to original resolution')
        os.system(f'gdalwarp -ts {outx} {outy} {self.small_mask_pth} {self.mask_pth}')
        print('Generating shape file')
#         os.system(f'gdal_polygonize.py {self.small_mask_pth} -f "ESRI Shapefile" {self.small_shape_pth}')
        os.system(f'gdal_polygonize.py {self.mask_pth} -f "ESRI Shapefile" {self.shape_pth}')
        return self.shape_pth

# model = MosaicSegmentationModel(model_pth = 'cdc-0.4-256.pkl', terrain_index=0, bs=24)
# terrain_detect = TerrainDetection(model, pixel_size=0.4, tile_size=256)
# mos_pth = '/home/rahul/raw_data/test_sites/mfar/vimana_photoscan-mosaic_utm_global_clipped_cmp.tif'
# out_folder = 'mfar_shape'
# res = terrain_detect.make_terrain_shape_file(mos_pth, out_folder, threshold=0.5)

if __name__ == "__main__":
    mos_pth = sys.argv[1]
    output_folder = sys.argv[2]
    
    model = MosaicSegmentationModel(model_pth = 'cdc-0.4-256.pkl', terrain_index=0, bs=24)
    terrain_detect = TerrainDetection(model, pixel_size=0.4, tile_size=256)
    res = terrain_detect.make_terrain_shape_file(mos_pth, output_folder, threshold=0.5)