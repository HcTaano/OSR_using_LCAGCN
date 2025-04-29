## 项目目的
使用LC-AGCN，结合openmax算法实现数据集开集识别。

## 数据集选用
### MSRAction3DSkeletonReal3D
MSR 3d action数据集记录了人体动作序列，共包含20个动作类型，10个被试者，每个被试者执行每个动作2或3次。总共有567个深度图序列。分辨率为640x240。
### UTKinect_skeletons（暂不训练）
The videos was captured using a single stationary Kinect with Kinect for Windows SDK Beta Version. There are 10 action types: walk, sit down, stand up, pick up, carry, throw, push, pull, wave hands, clap hands. There are 10 subjects, Each subject performs each actions twice. Three channels were recorded: RGB, depth and skeleton joint locations. The three channel are synchronized. The framerate is 30f/s. Note we only recorded the frames when the skeleton was tracked, the frame number of the files has jumps. The final frame rate is about 15f/sec. (There is around 2% of the frames where there are multiple skeleton info recorded with slightly different joint locations. This is not caused by a second person. You can chooce either one. )
Sketetal joint Locations (.txt) Each row contains the data of one frame, the first number is frame number, the following numbers are the (x,y,z) locations of joint 1-20. The x, y, and z are the coordinates relative to the sensor array, in meters. Detailed description of the coordinates can be found here The index of the joints are described here.

## 当前存在的问题
1. 我当前不想解析utk的标签，因为它会让openmax_inference.py出现错误
2. 在epoch==1000,lr==0.0001,batch_size==128下，MSR的开集AUC为0.563，极度需要优化（最重要）
3. 让openmax_inference.py输出更多的评价指标，另外最好也要加入闭集的指标性能检测。
4. 让一直类别的数量适当增大（如增大到18类），需要修改对应代码。


## 电脑性能
GPU: 3060laptop
CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz   2.30 GHz
内存: 16GB
版本: Windows 11 专业版

## 当前依赖环境
### 环境名称：lcagcn_osr310
### 软件包
  Name                    Version                   Build  Channel
blas                      1.0                         mkl  
brotli                    1.1.0                h2466b09_2    conda-forge
brotli-bin                1.1.0                h2466b09_2    conda-forge
brotli-python             1.0.9           py310h5da7b33_9  
bzip2                     1.0.8                h2bbff1b_6  
ca-certificates           2025.4.26            h4c7d964_0    conda-forge
certifi                   2025.1.31          pyhd8ed1ab_0    conda-forge
charset-normalizer        3.3.2              pyhd3eb1b0_0
contourpy                 1.3.2           py310hc19bc0b_0    conda-forge
cuda-cccl                 12.8.90                       0    nvidia
cuda-cccl_win-64          12.8.90                       0    nvidia
cuda-cudart               11.8.89                       0    nvidia
cuda-cudart-dev           11.8.89                       0    nvidia
cuda-cupti                11.8.87                       0    nvidia
cuda-libraries            11.8.0                        0    nvidia
cuda-libraries-dev        11.8.0                        0    nvidia
cuda-nvrtc                11.8.89                       0    nvidia
cuda-nvrtc-dev            11.8.89                       0    nvidia
cuda-nvtx                 11.8.86                       0    nvidia
cuda-profiler-api         12.8.90                       0    nvidia
cuda-runtime              11.8.0                        0    nvidia
cuda-version              12.8                          3    nvidia
cycler                    0.12.1             pyhd8ed1ab_1    conda-forge
cython                    3.0.12          py310h6bd2d47_0    conda-forge
filelock                  3.17.0          py310haa95532_0
fonttools                 4.57.0          py310h38315fa_0    conda-forge
freeglut                  3.4.0                hd77b12b_0
freetype                  2.13.3               h0620614_0
gmp                       6.3.0                h537511b_0
gmpy2                     2.2.1           py310h827c3e9_0
icu                       73.2                 h63175ca_0    conda-forge
idna                      3.7             py310haa95532_0
intel-openmp              2023.1.0         h59b6b97_46320
jinja2                    3.1.6           py310haa95532_0
joblib                    1.4.2              pyhd8ed1ab_1    conda-forge
jpeg                      9e                   h827c3e9_3
kiwisolver                1.4.7           py310hc19bc0b_0    conda-forge
krb5                      1.21.3               hdf4eb48_0    conda-forge
lcms2                     2.16                 h62be587_1
lerc                      4.0.0                h5da7b33_0
libblas                   3.9.0           1_h8933c1f_netlib    conda-forge
libbrotlicommon           1.1.0                h2466b09_2    conda-forge
libbrotlidec              1.1.0                h2466b09_2    conda-forge
libbrotlienc              1.1.0                h2466b09_2    conda-forge
libcblas                  3.9.0           8_h719fc58_netlib    conda-forge
libclang13                14.0.6          default_h8e68704_2
libcublas                 11.11.3.6                     0    nvidia
libcublas-dev             11.11.3.6                     0    nvidia
libcufft                  10.9.0.58                     0    nvidia
libcufft-dev              10.9.0.58                     0    nvidia
libcurand                 10.3.9.90                     0    nvidia
libcurand-dev             10.3.9.90                     0    nvidia
libcusolver               11.4.1.48                     0    nvidia
libcusolver-dev           11.4.1.48                     0    nvidia
libcusparse               11.7.5.86                     0    nvidia
libcusparse-dev           11.7.5.86                     0    nvidia
libdeflate                1.22                 h5bf469e_0
libffi                    3.4.4                hd77b12b_1
libiconv                  1.18                 h135ad9c_1    conda-forge
libjpeg-turbo             2.0.0                h196d8e1_0
liblapack                 3.9.0           8_h719fc58_netlib    conda-forge
libmr                     0.1.9                    pypi_0    pypi
libnpp                    11.8.0.86                     0    nvidia
libnpp-dev                11.8.0.86                     0    nvidia
libnvjpeg                 11.9.0.86                     0    nvidia
libnvjpeg-dev             11.9.0.86                     0    nvidia
libpng                    1.6.39               h8cc25b3_0
libpq                     17.0                 hfc46ca6_0    conda-forge
libtiff                   4.7.0                h404307b_0
libuv                     1.48.0               h827c3e9_0
libwebp-base              1.3.2                h3d04722_1
libxml2                   2.13.7               h866ff63_0
libxslt                   1.1.41               h0739af5_0
lz4-c                     1.9.4                h2bbff1b_1
m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
m2w64-gcc-libs            5.3.0                         7    conda-forge
m2w64-gcc-libs-core       5.3.0                         7    conda-forge
m2w64-gmp                 6.1.0                         2    conda-forge
m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
markupsafe                3.0.2           py310h827c3e9_0
matplotlib                3.10.1          py310h5588dad_0    conda-forge
matplotlib-base           3.10.1          py310h37e0a56_0    conda-forge
minizip                   4.0.3                hb68bac4_0
mkl                       2023.1.0         h6b88ed4_46358
mkl-service               2.4.0           py310h827c3e9_2
mkl_fft                   1.3.11          py310h827c3e9_0
mkl_random                1.2.8           py310hc64d2fc_0
mpc                       1.3.1                h827c3e9_0
mpfr                      4.2.1                h56c3642_0
mpmath                    1.3.0           py310haa95532_0
msys2-conda-epoch         20160418                      1    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
networkx                  3.4.2              pyh267e887_2    conda-forge
numpy                     2.0.1           py310h055cbcc_1
numpy-base                2.0.1           py310h65a83cf_1
openjpeg                  2.5.2                h9b5d1b5_1
openssl                   3.5.0                ha4e3fda_0    conda-forge
packaging                 25.0               pyh29332c3_1    conda-forge
pillow                    11.1.0          py310hea0d53e_1
pip                       25.0            py310haa95532_0
pyparsing                 3.2.3              pyhd8ed1ab_1    conda-forge
pyside6                   6.7.2           py310h5ef65bb_0
pysocks                   1.7.1           py310haa95532_0
python                    3.10.16              h4607a30_1
python-dateutil           2.9.0.post0        pyhff2d567_1    conda-forge
python_abi                3.10                    2_cp310    conda-forge
pytorch                   2.1.0           py3.10_cuda11.8_cudnn8_0    pytorch
pytorch-cuda              11.8                 h24eeafa_6    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pyyaml                    6.0.2           py310h827c3e9_0
qhull                     2020.2               hc790b64_5    conda-forge
qtbase                    6.7.3                h0804d20_0
qtdeclarative             6.7.3                h5da7b33_0
qtshadertools             6.7.3                h5da7b33_0
qtsvg                     6.7.3                hf2fb9eb_0
qttools                   6.7.3                h0de5f00_0
qtwebchannel              6.7.3                h5da7b33_0
qtwebengine               6.7.3                hc184ec4_0
qtwebsockets              6.7.3                h5da7b33_0
requests                  2.32.3          py310haa95532_1
scikit-learn              1.6.1           py310hf2a6c47_0    conda-forge
scipy                     1.15.2          py310h15c175c_0    conda-forge
setuptools                75.8.0          py310haa95532_0
six                       1.17.0             pyhd8ed1ab_0    conda-forge
sqlite                    3.45.3               h2bbff1b_0
sympy                     1.13.3          py310haa95532_1
tbb                       2021.8.0             h59b6b97_0
threadpoolctl             3.6.0              pyhecae5ae_0    conda-forge
tk                        8.6.14               h0416ee5_0
torchaudio                2.1.0                    pypi_0    pypi
torchvision               0.16.0                   pypi_0    pypi
tornado                   6.4.2           py310ha8f682b_0    conda-forge
typing_extensions         4.12.2          py310haa95532_0
tzdata                    2025a                h04d1e81_0
ucrt                      10.0.22621.0         h57928b3_1    conda-forge
unicodedata2              16.0.0          py310ha8f682b_0    conda-forge
urllib3                   2.3.0           py310haa95532_0
vc                        14.42                haa95532_5
vc14_runtime              14.42.34438         hfd919c2_26    conda-forge
vs2015_runtime            14.42.34438         h7142326_26    conda-forge
wheel                     0.45.1          py310haa95532_0
win_inet_pton             1.1.0           py310haa95532_0
xz                        5.6.4                h4754444_1
yaml                      0.2.5                he774522_0
zlib                      1.2.13               h8cc25b3_1
zstd                      1.5.6                h8880b57_0
