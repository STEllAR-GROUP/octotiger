
cmake -DCMAKE_PREFIX_PATH=/usr/local/packages/hpx/ $1  \
      -DCMAKE_CXX_COMPILER="g++"                                              \
      -DCMAKE_C_COMPILER="gcc"                                                 \
      -DCMAKE_C_FLAGS="-O2 -g"                                \
      -DCMAKE_CXX_FLAGS="-O2 -g"                                


