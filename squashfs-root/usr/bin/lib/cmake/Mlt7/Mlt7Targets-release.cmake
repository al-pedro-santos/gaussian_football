#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Mlt7::mlt" for configuration "Release"
set_property(TARGET Mlt7::mlt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Mlt7::mlt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmlt-7.so.7.39.0"
  IMPORTED_SONAME_RELEASE "libmlt-7.so.7"
  )

list(APPEND _IMPORT_CHECK_TARGETS Mlt7::mlt )
list(APPEND _IMPORT_CHECK_FILES_FOR_Mlt7::mlt "${_IMPORT_PREFIX}/lib/libmlt-7.so.7.39.0" )

# Import target "Mlt7::mlt++" for configuration "Release"
set_property(TARGET Mlt7::mlt++ APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Mlt7::mlt++ PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmlt++-7.so.7.39.0"
  IMPORTED_SONAME_RELEASE "libmlt++-7.so.7"
  )

list(APPEND _IMPORT_CHECK_TARGETS Mlt7::mlt++ )
list(APPEND _IMPORT_CHECK_FILES_FOR_Mlt7::mlt++ "${_IMPORT_PREFIX}/lib/libmlt++-7.so.7.39.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
