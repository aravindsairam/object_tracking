cmake_minimum_required(VERSION 3.18)
project(object_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen objects (Eigen::MatrixXf) cross the boundary between motcpp and our code
# (ByteTrack::update returns one, we free it). If the two are compiled with
# different Eigen alignment, the aligned malloc/free ABI mismatches -> heap
# corruption at high track counts. Force a single, consistent setting for ALL
# targets below (motcpp included, since it's add_subdirectory'd into this build).
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=32 EIGEN_MAX_STATIC_ALIGN_BYTES=32)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# FFmpeg libraries for the NVDEC (NvdecDecoder) hardware-decode path. The system
# libavcodec is built with NVIDIA cuvid (it links libnvcuvid at runtime), so no
# extra SDK is needed. Used directly (NOT via OpenCV's VideoCapture).
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat libavcodec libavutil libswscale)

# ---- ONNX Runtime (vendored under third_party, GPU build) --------------
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
if(NOT EXISTS ${ORT_ROOT}/include/onnxruntime_cxx_api.h)
  message(FATAL_ERROR "ONNX Runtime not found at ${ORT_ROOT} — see third_party/onnxruntime")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_ROOT}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include)

# CUDA toolkit lib dir (cuDNN 9 is bundled into ${ORT_ROOT}/lib alongside the
# ORT provider libs). These dirs go on the executable's RPATH below.
set(CUDA_LIB_DIR /usr/local/cuda-12.6/targets/x86_64-linux/lib)

# TensorRT 10 runtime libs. ORT's TensorRT EP dlopen()s libnvinfer.so.10 and
# libnvonnxparser.so.10; if they are not on the (DT_RPATH) search path the EP
# silently falls back to the much slower CUDA EP (fp16 is ignored there). These
# ship in the pip 'tensorrt' wheel under site-packages/tensorrt_libs. Override
# with -DTENSORRT_LIB_DIR=/path if your install differs.
set(TENSORRT_LIB_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/tensorrt_libs"
    CACHE PATH "Directory containing libnvinfer.so.10 (TensorRT 10 runtime)")
if(NOT EXISTS "${TENSORRT_LIB_DIR}/libnvinfer.so.10")
  message(WARNING "libnvinfer.so.10 not found in TENSORRT_LIB_DIR='${TENSORRT_LIB_DIR}'. "
                  "The TensorRT EP will fall back to the slower CUDA EP. "
                  "Install it (pip install tensorrt) or pass -DTENSORRT_LIB_DIR=...")
endif()

# ---- motcpp (vendored, AGPL-3.0; isolated behind src/track/mot_tracker.cpp) --
# Build only the library: no tests/examples/tools/install, ONNX off until M5.
set(MOTCPP_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(MOTCPP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MOTCPP_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(MOTCPP_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(MOTCPP_INSTALL          OFF CACHE BOOL "" FORCE)
set(MOTCPP_ENABLE_ONNX      OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/motcpp EXCLUDE_FROM_ALL)

# ---- core library (grows each milestone) -------------------------------
add_library(ot_core
  src/io/video_source.cpp
  src/io/ocv_decoder.cpp
  src/io/nvdec_decoder.cpp
  src/core/config.cpp
  src/detect/class_map.cpp
  src/detect/yolo_detector.cpp
  src/detect/rfdetr_detector.cpp
  src/detect/ort_backend.cpp
  src/detect/tiled_detector.cpp
  src/detect/detector_factory.cpp
  src/track/mot_tracker.cpp
  src/lock/lock_manager.cpp
  src/reid/histogram_embedder.cpp
  src/reid/onnx_reid_embedder.cpp
  src/output/jsonl_sink.cpp
  src/ui/overlay.cpp
  src/ui/selector.cpp
)
target_include_directories(ot_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
# src/detect/*.cpp include their own private headers from src/detect.
target_include_directories(ot_core PRIVATE ${CMAKE_SOURCE_DIR}/src/detect)

# Tune the CPU-side hot loops (RF-DETR per-channel normalize + per-tile decode)
# for this machine. -march=native ties the binary to the build host's CPU, which
# is fine for a fixed deployment target. Opt-out with -DOT_NATIVE_ARCH=OFF.
option(OT_NATIVE_ARCH "Build with -march=native -funroll-loops" ON)
if(OT_NATIVE_ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ot_core PRIVATE
    $<$<CONFIG:Release>:-march=native -funroll-loops>)
endif()
# motcpp is a private impl detail (used only inside src/track/mot_tracker.cpp).
target_link_libraries(ot_core PUBLIC ${OpenCV_LIBS} yaml-cpp onnxruntime PkgConfig::FFMPEG)
target_link_libraries(ot_core PRIVATE motcpp)

# Embed an rpath so the binary finds libonnxruntime.so without LD_LIBRARY_PATH.
set_target_properties(ot_core PROPERTIES INSTALL_RPATH ${ORT_ROOT}/lib)

# ---- application -------------------------------------------------------
add_executable(object_tracking src/app/main.cpp)
target_link_libraries(object_tracking PRIVATE ot_core)
# Use OLD dtags (DT_RPATH, not DT_RUNPATH): DT_RPATH is consulted for ALL library
# lookups in the process, so ORT's dlopen()ed CUDA provider finds the bundled
# cuDNN in ${ORT_ROOT}/lib without needing LD_LIBRARY_PATH.
target_link_options(object_tracking PRIVATE -Wl,--disable-new-dtags)
set_target_properties(object_tracking PROPERTIES
  BUILD_RPATH   "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}")

# ---- headless smoke / parity tools -----------------------------------------
add_executable(rfdetr_smoke tools/rfdetr_smoke.cpp)
target_link_libraries(rfdetr_smoke PRIVATE ot_core)
target_link_options(rfdetr_smoke PRIVATE -Wl,--disable-new-dtags)
set_target_properties(rfdetr_smoke PROPERTIES
  BUILD_RPATH   "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}")

add_executable(reid_smoke tools/reid_smoke.cpp)
target_link_libraries(reid_smoke PRIVATE ot_core)
target_link_options(reid_smoke PRIVATE -Wl,--disable-new-dtags)
set_target_properties(reid_smoke PROPERTIES
  BUILD_RPATH   "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}")

# Logic-only unit test for LockManager appearance verification (no GPU/model).
add_executable(test_lock_appearance tools/test_lock_appearance.cpp)
target_link_libraries(test_lock_appearance PRIVATE ot_core)
target_link_options(test_lock_appearance PRIVATE -Wl,--disable-new-dtags)
set_target_properties(test_lock_appearance PROPERTIES
  BUILD_RPATH   "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "${ORT_ROOT}/lib;${CUDA_LIB_DIR};${TENSORRT_LIB_DIR}")

# ---- warnings on our own targets (vendored motcpp left untouched) ----------
# -Wall -Wextra catches dead functions, unused vars/params, sign-compare slips.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(tgt ot_core object_tracking rfdetr_smoke reid_smoke test_lock_appearance)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra)
  endforeach()
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
