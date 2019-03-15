# terrain_detection

`conda activate fastaiv1`
`python make_terrain_shape.py {mosaic_pth} {output_folder}`

mosaic_pth = Path to the mosaic image
output_folder = folder where shape files will be stored

### example:
`python make_terrain_shape.py raw_data/sakleshpur/mosaic.tif raw_data/sakleshpur/shape`

This will generate the shape files for `raw_data/sakleshpur/mosaic.tif` and save it in `raw_data/sakleshpur/shape`

files in `raw_data/sakleshpur/shape`:
mosaic_shape.shp
mosaic_shape.prj
mosaic_shape.dbf
mosaic_shape.shx