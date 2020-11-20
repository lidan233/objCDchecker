# CMake generated Testfile for 
# Source directory: C:/Users/lidan/Desktop/last/externel/cxxopts/test
# Build directory: C:/Users/lidan/Desktop/last/cmake-build-release/externel/cxxopts/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(options "options_test")
set_tests_properties(options PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/lidan/Desktop/last/externel/cxxopts/test/CMakeLists.txt;4;add_test;C:/Users/lidan/Desktop/last/externel/cxxopts/test/CMakeLists.txt;0;")
add_test(find-package-test "D:/software/clion/CLion 2020.2.3/bin/cmake/win/bin/ctest.exe" "-C" "Release" "--build-and-test" "C:/Users/lidan/Desktop/last/externel/cxxopts/test/find-package-test" "C:/Users/lidan/Desktop/last/cmake-build-release/externel/cxxopts/test/find-package-test" "--build-generator" "NMake Makefiles" "--build-makeprogram" "nmake" "--build-options" "-DCMAKE_CXX_COMPILER=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.27.29110/bin/Hostx64/x64/cl.exe" "-DCMAKE_BUILD_TYPE=Release" "-Dcxxopts_DIR=C:/Users/lidan/Desktop/last/cmake-build-release/externel/cxxopts")
set_tests_properties(find-package-test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/lidan/Desktop/last/externel/cxxopts/test/CMakeLists.txt;7;add_test;C:/Users/lidan/Desktop/last/externel/cxxopts/test/CMakeLists.txt;0;")
add_test(add-subdirectory-test "D:/software/clion/CLion 2020.2.3/bin/cmake/win/bin/ctest.exe" "-C" "Release" "--build-and-test" "C:/Users/lidan/Desktop/last/externel/cxxopts/test/add-subdirectory-test" "C:/Users/lidan/Desktop/last/cmake-build-release/externel/cxxopts/test/add-subdirectory-test" "--build-generator" "NMake Makefiles" "--build-makeprogram" "nmake" "--build-options" "-DCMAKE_CXX_COMPILER=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.27.29110/bin/Hostx64/x64/cl.exe" "-DCMAKE_BUILD_TYPE=Release")
set_tests_properties(add-subdirectory-test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/lidan/Desktop/last/externel/cxxopts/test/CMakeLists.txt;21;add_test;C:/Users/lidan/Desktop/last/externel/cxxopts/test/CMakeLists.txt;0;")
