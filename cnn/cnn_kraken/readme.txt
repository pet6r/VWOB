To rebuild the build folder and make the project, run the following commands:

cd /home/ptr/Documents/vwob/cnn/cnn_kraken

rm -rf build

mkdir build && cd build

cmake .. -DPYTHON_EXECUTABLE=$(which python3)

make
