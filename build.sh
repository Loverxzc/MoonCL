#!/bin/bash
mkdir build
i686-w64-mingw32-g++ -shared -m32 -O3 \
          src/*.cpp -o build/mooncl.dll \
          -I include \
          lib32/OpenCL.dll \
          -static -static-libgcc -static-libstdc++ \
          -Wl,--enable-stdcall-fixup