set -x

rm -rf $1
mkdir $1
cd $1
rm CMakeCache.txt
rm -r CMakeFiles

cmake                                                        \
      	-DCMAKE_BUILD_TYPE=$1                                \
     	-DCPPuddle_DIR=$HOME/local/debug/lib/cmake/CPPuddle/ \
      	-DHPX_DIR=$HOME/local/debug/lib/cmake/HPX/           \
        -DOCTOTIGER_WITH_TESTS=ON                            \
      	-DVc_DIR=$HOME/local/lib/cmake/Vc/                   \
        ..


make -j


