import numpy as np
import mazalib

side_size = 300
im = np.fromfile('image3d.raw', dtype='uint8')
im = np.reshape(im, (side_size,side_size,side_size))


# 300 300 300
# 3 m i v
# 23 56

# width height depth
# Radius, VarMethod, CorMethod, OutFormat
# thresholdLow thresholdHigh

print(" --- Start kriging!")
result=mazalib.kriging(im, 2, (23, 56))
print(" --- Result kriging!")



# # 300 300 300
# # 1.01 1.02 1.0 1.0
# # 1 10
# # 23 56

# # Annotation:
# # width height depth
# # alphaG alphaI G0 unsharp_mask_strength
# # nlm_iterations nlm_search_radius
# # low_threshold high_threshold

print(" --- Start cac!")
result=mazalib.cac(im,(1.02, 1.03, 1.01),(23, 56))
print(" --- Result cac!")



# 300 300 300
# 0.5 0.98 0 1.0 0.5 0.001 20
# 23 56

# width height depth
# Beta Speed Method(0=MMD,1=ICM,2=None) tStart Alpha EnergyThreshold MaxIterations
# thresholdLow thresholdHigh

print(" --- Start mrf!")
result=mazalib.mrf(im,(0.5, 0.98, 0, 1.0, 0.5, 0.001, 20),(23, 56))
print(" --- Result mrf!")



# # 300 300 300
# # 23 56

# # width height depth
# # thresholdLow thresholdHigh

print(" --- Start rgs!")
result=mazalib.rgs(im,(23, 56))
print(" --- Result rgs!")



# 300 300 300
# 1 50 1.0 1 1 1 0

# width height depth
# order threshold gain sgn nScales is_auto use_scales
# thresholdLow thresholdHigh

print(" --- Start hessian!")
result=mazalib.hessian(im,(1, 50, 1.0, 1, 1, 1, 0))
print(" --- Result hessian!")



# 300 300 300
# 1 50 1.0 1 1 1 0
# 1.0 2 1 2 1 1.0
# 23 56

# width height depth
# order threshold gain sgn nScales is_auto use_scales
# FreezingSpeed HessianOrder nSteps Radius Strength TStart
# lowThreshold highThreshold

print(" --- Start windowedHessian!")
result=mazalib.windowedHessian(im,(1, 50, 1.0, 1, 1, 1, 0),(1.0, 2, 1, 2, 1, 1.0),(23, 56))
print(" --- Result windowedHessian!")
