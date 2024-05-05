import numpy as np
import mazalib as testedLib


im=np.fromfile('../MAZAlib_v1.0/input_data_and_config_files/image3d.raw', dtype='uint8')

sideSize=int(round(im.size**(1/3)))
im=np.reshape(im,(sideSize,sideSize,sideSize))

print(im)
print(sideSize)

localIm=im


# 300 300 300
# 3 m i v
# 23 56
# image3d.raw

# width height depth
# Radius, VarMethod, CorMethod, OutFormat
# thresholdLow thresholdHigh
# outputImageFile


# result=testedLib.kriging(localIm,( 'm', 'i', 'v', 3 ),(23, 56))
# print(" --- Result kriging!")



# 300 300 300
# 0.5 0.98 0 1.0 0.5 0.001 20
# 23 56
# image3d.raw

# width height depth
# Beta Speed Method(0=MMD,1=ICM,2=None) tStart Alpha EnergyThreshold MaxIterations
# thresholdLow thresholdHigh
# inputImageFile

# print(" --- Start mrf!")
# result=testedLib.mrf(localIm,(0.5, 0.98, 0, 1.0, 0.5, 0.001, 20),(23, 56))
# print(" --- Result mrf!")

# # 300 300 300
# # 23 56
# # image3d.raw

# # width height depth
# # thresholdLow thresholdHigh
# # outputImageFile


# print(" --- Start rgs!")
# result=testedLib.rgs(localIm,(23, 56))
# print(" --- Result rgs!")

# 300 300 300
# 1 50 1.0 1 1 1 0
# image3d.raw

# width height depth
# order threshold gain sgn nScales is_auto use_scales
# thresholdLow thresholdHigh
# outputImageFile


# print(" --- Start hessian!")
# result=testedLib.hessian(localIm,(1, 50, 1.0, 1, 1, 1, 0))
# print(" --- Result hessian!")

# 300 300 300
# 1 50 1.0 1 1 1 0
# 1.0 2 1 2 1 1.0
# 23 56
# image3d.raw

# width height depth
# order threshold gain sgn nScales is_auto use_scales
# FreezingSpeed HessianOrder nSteps Radius Strength TStart
# lowThreshold highThreshold
# outputImageFile

# print(" --- Start windowedHessian!")
# result=testedLib.windowedHessian(localIm,(1, 50, 1.0, 1, 1, 1, 0),(1.0, 2, 1, 2, 1, 1.0),(23, 56))
# print(" --- Result windowedHessian!")


print(" --- Start unsharp!")
result=testedLib.unsharp(localIm,[3.0])
print(" --- Result unsharp!")

print(" --- Start nlm!")
result=testedLib.nlm(localIm,(2,3))
print(" --- Result nlm!")

# # 300 300 300
# # 1.01 1.02 1.0 1.0
# # 1 10
# # 23 56
# # image3d.raw

# # Annotation:
# # width height depth
# # alphaG alphaI G0 unsharp_mask_strength
# # nlm_iterations nlm_search_radius
# # low_threshold high_threshold
# # output_image_filename
# # was 23, 56
print(" --- Start cac!")
result=testedLib.cac(localIm,(1.02, 1.03, 1.01),(23, 56))
print(" --- Result cac!")

