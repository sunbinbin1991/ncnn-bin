mkdir build_android  
cd build_android  
cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE="..\android.toolchain.cmake" ..\ -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="armeabi-v7a" -DANDROID_NATIVE_API_LEVEL=9  
cmake --build .  
cmake --build . --target install  
cd ..  
pause