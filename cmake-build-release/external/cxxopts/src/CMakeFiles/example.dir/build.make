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
include external\cxxopts\src\CMakeFiles\example.dir\depend.make

# Include the progress variables for this target.
include external\cxxopts\src\CMakeFiles\example.dir\progress.make

# Include the compile flags for this target's objects.
include external\cxxopts\src\CMakeFiles\example.dir\flags.make

external\cxxopts\src\CMakeFiles\example.dir\example.cpp.obj: external\cxxopts\src\CMakeFiles\example.dir\flags.make
external\cxxopts\src\CMakeFiles\example.dir\example.cpp.obj: ..\external\cxxopts\src\example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\lidan\Desktop\last\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/cxxopts/src/CMakeFiles/example.dir/example.cpp.obj"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\example.dir\example.cpp.obj /FdCMakeFiles\example.dir\ /FS -c C:\Users\lidan\Desktop\last\external\cxxopts\src\example.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

external\cxxopts\src\CMakeFiles\example.dir\example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/example.cpp.i"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe > CMakeFiles\example.dir\example.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\lidan\Desktop\last\external\cxxopts\src\example.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

external\cxxopts\src\CMakeFiles\example.dir\example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/example.cpp.s"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\example.dir\example.cpp.s /c C:\Users\lidan\Desktop\last\external\cxxopts\src\example.cpp
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

# Object files for target example
example_OBJECTS = \
"CMakeFiles\example.dir\example.cpp.obj"

# External object files for target example
example_EXTERNAL_OBJECTS =

external\cxxopts\src\example.exe: external\cxxopts\src\CMakeFiles\example.dir\example.cpp.obj
external\cxxopts\src\example.exe: external\cxxopts\src\CMakeFiles\example.dir\build.make
external\cxxopts\src\example.exe: external\cxxopts\src\CMakeFiles\example.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\lidan\Desktop\last\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example.exe"
	cd C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src
	"D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\example.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\example.dir\objects1.rsp @<<
 /out:example.exe /implib:example.lib /pdb:C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src\example.pdb /version:0.0  /machine:x64 /INCREMENTAL:NO /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<
	cd C:\Users\lidan\Desktop\last\cmake-build-release

# Rule to build all files generated by this target.
external\cxxopts\src\CMakeFiles\example.dir\build: external\cxxopts\src\example.exe

.PHONY : external\cxxopts\src\CMakeFiles\example.dir\build

external\cxxopts\src\CMakeFiles\example.dir\clean:
	cd C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src
	$(CMAKE_COMMAND) -P CMakeFiles\example.dir\cmake_clean.cmake
	cd C:\Users\lidan\Desktop\last\cmake-build-release
.PHONY : external\cxxopts\src\CMakeFiles\example.dir\clean

external\cxxopts\src\CMakeFiles\example.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\lidan\Desktop\last C:\Users\lidan\Desktop\last\external\cxxopts\src C:\Users\lidan\Desktop\last\cmake-build-release C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src C:\Users\lidan\Desktop\last\cmake-build-release\external\cxxopts\src\CMakeFiles\example.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : external\cxxopts\src\CMakeFiles\example.dir\depend

