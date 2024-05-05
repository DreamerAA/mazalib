
set(SOURCES_LIB
    src/CACSegmentation.cpp
    src/Grid.cpp
    src/HessianSegmentation.cpp
    src/IntegralImage.cpp
    src/KrigingSegmentation.cpp
    src/LogFile.cpp
    src/MRFSegmentation.cpp
    src/NewCACSegmentation.cpp
    src/RandomNumbers.cpp
    src/threshold.cpp
    src/Variation.cpp
    src/IterativeNonLocalMeansCPU.cpp
    src/NoiseLevel.cpp 
    src/NonLocalMeans.cpp  
    src/LatticeModel.cpp  
    src/ExpectedMaximized.cpp
)

add_library(segmentation SHARED ${SOURCES_LIB})
target_link_libraries(segmentation PUBLIC Eigen3::Eigen)
target_include_directories(segmentation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
