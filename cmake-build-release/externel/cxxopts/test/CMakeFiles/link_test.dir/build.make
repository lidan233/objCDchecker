# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\lidan\Desktop\last

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\lidan\Desktop\last\cmake-build-release

# Include any dependencies generated for this target.
include externel\cxxopts\test\CMakeFiles\link_test.dir\depend.make

# Include the progress variables for this target.
include externel\cxxopts\test\CMakeFiles\link_test.dir\progress.make

# Include the compile flags for this target's objects.
include externel\cxxopts\test\CMakeFiles\link_test.dir\flags.make

externel\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.obj: externel\cxxopts\test\CMakeFiles\link_test.dir\flags.make
externel\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.obj: ..\externel\cxxopts\test\link_a.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\lidan\Desktop\last\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object externel/cxxopts/test/CMakeFiles/link_test.dir/link_a.cpp.obj"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\link_test.dir\link_a.cpp.obj /FdCMakeFiles\link_test.dir\ /FS -c C:\Users\lidan\Desktop\last\externel\cxxopts\test\link_a.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

externel\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/link_test.dir/link_a.cpp.i"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe > CMakeFiles\link_test.dir\link_a.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\lidan\Desktop\last\externel\cxxopts\test\link_a.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

externel\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/link_test.dir/link_a.cpp.s"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\link_test.dir\link_a.cpp.s /c C:\Users\lidan\Desktop\last\externel\cxxopts\test\link_a.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

externel\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.obj: externel\cxxopts\test\CMakeFiles\link_test.dir\flags.make
externel\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.obj: ..\externel\cxxopts\test\link_b.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\lidan\Desktop\last\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object externel/cxxopts/test/CMakeFiles/link_test.dir/link_b.cpp.obj"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\link_test.dir\link_b.cpp.obj /FdCMakeFiles\link_test.dir\ /FS -c C:\Users\lidan\Desktop\last\externel\cxxopts\test\link_b.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

externel\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/link_test.dir/link_b.cpp.i"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe > CMakeFiles\link_test.dir\link_b.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\lidan\Desktop\last\externel\cxxopts\test\link_b.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

externel\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/link_test.dir/link_b.cpp.s"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\link_test.dir\link_b.cpp.s /c C:\Users\lidan\Desktop\last\externel\cxxopts\test\link_b.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

# Object files for target link_test
link_test_OBJECTS = \
"CMakeFiles\link_test.dir\link_a.cpp.obj" \
"CMakeFiles\link_test.dir\link_b.cpp.obj"

# External object files for target link_test
link_test_EXTERNAL_OBJECTS =

externel\cxxopts\test\link_test.exe: externel\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.obj
externel\cxxopts\test\link_test.exe: externel\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.obj
externel\cxxopts\test\link_test.exe: externel\cxxopts\test\CMakeFiles\link_test.dir\build.make
externel\cxxopts\test\link_test.exe: externel\cxxopts\test\CMakeFiles\link_test.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\lidan\Desktop\last\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable link_test.exe"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	"D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\link_test.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\link_test.dir\objects1.rsp @<<
 /out:link_test.exe /implib:link_test.lib /pdb:C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test\link_test.pdb /version:0.0  /machine:x64 /INCREMENTAL:NO /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

# Rule to build all files generated by this target.
externel\cxxopts\test\CMakeFiles\link_test.dir\build: externel\cxxopts\test\link_test.exe

.PHONY : externel\cxxopts\test\CMakeFiles\link_test.dir\build

externel\cxxopts\test\CMakeFiles\link_test.dir\clean:
	cd C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test
	$(CMAKE_COMMAND) -P CMakeFiles\link_test.dir\cmake_clean.cmake
	cd C:\Users\lidan\Desktop\last\cmake-build-release
.PHONY : externel\cxxopts\test\CMakeFiles\link_test.dir\clean

externel\cxxopts\test\CMakeFiles\link_test.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\lidan\Desktop\last C:\Users\lidan\Desktop\last\externel\cxxopts\test C:\Users\lidan\Desktop\last\cmake-build-release C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test C:\Users\lidan\Desktop\last\cmake-build-release\externel\cxxopts\test\CMakeFiles\link_test.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : externel\cxxopts\test\CMakeFiles\link_test.dir\depend
