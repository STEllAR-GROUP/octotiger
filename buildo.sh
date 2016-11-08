
cmake -DCMAKE_PREFIX_PATH=/work/$USER/release/hpx $1  \
      -DCMAKE_CXX_COMPILER="/usr/local/compilers/Intel/parallel_studio_xe_2015/impi/5.0.1.035/intel64/bin/mpiicpc"                                              \
      -DCMAKE_C_COMPILER="/usr/local/compilers/Intel/parallel_studio_xe_2015/impi/5.0.1.035/intel64/bin/mpiicc"                                                 \
      -DCMAKE_AR="/usr/local/compilers/Intel/parallel_studio_xe_2015/composer_xe_2015.0.090/bin/intel64/xiar"\
      -DCMAKE_BUILD_TYPE=release                                                                                                                                \
      -DSilo_LIBRARY="$HOME/lib/libsilo.a" -DSilo_INCLUDE_DIR="$HOME/include" \
      -DCMAKE_C_FLAGS="-xAVX -mt_mpi -gcc-name=gcc  -O3 -g3 -ip"                                \
      -DCMAKE_CXX_FLAGS="-xAVX -std=c++11 -diag-disable=488 -mt_mpi -gxx-name=g++ -O3 -g3 -ip"                                

