import os
import subprocess
import numpy as np
from osgeo import gdal


def _16bit_to_8Bit(inputRaster, outputRaster, outputPixType='Byte', outputFormat='png', percentiles=[2, 98]):
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of',
           outputFormat]

    for bandId in range(srcRaster.RasterCount):
        bandId = bandId + 1
        band = srcRaster.GetRasterBand(bandId)

        bmin = band.GetMinimum()
        bmax = band.GetMaximum()
        # if not exist minimum and maximum values
        if bmin is None or bmax is None:
            (bmin, bmax) = band.ComputeRasterMinMax(1)
        # else, rescale
        band_arr_tmp = band.ReadAsArray()
        bmin = np.percentile(band_arr_tmp.flatten(),
                             percentiles[0])
        bmax = np.percentile(band_arr_tmp.flatten(),
                             percentiles[1])

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print("Conversin command:", cmd)
    subprocess.call(cmd)


path = '../AerialImageDataset'
path1 = path + "/test/train/"
files = os.listdir(path1)

for file in files:
    resimPath = path1 + file
    dstPath = path + "rescale/" + file[:-3] + "png"

    _16bit_to_8Bit(resimPath, dstPath)
    pass

gdal.UseExceptions()

# ds = gdal.Open('./SN3_roads_train_AOI_3_Paris_PS-RGB_img250.tif')
# band1 = ds.GetRasterBand(1).ReadAsArray()
# band2 = ds.GetRasterBand(2).ReadAsArray()
# band3 = ds.GetRasterBand(3).ReadAsArray()
# rgb_array = np.dstack([band3,band2,band1])
# plt.imshow(rgb_array)
# plt.show()
# pass
