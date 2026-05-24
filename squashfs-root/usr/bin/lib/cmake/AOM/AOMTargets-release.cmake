#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "AOM::aom" for configuration "Release"
set_property(TARGET AOM::aom APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AOM::aom PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaom.so.3.13.1"
  IMPORTED_SONAME_RELEASE "libaom.so.3"
  )

list(APPEND _IMPORT_CHECK_TARGETS AOM::aom )
list(APPEND _IMPORT_CHECK_FILES_FOR_AOM::aom "${_IMPORT_PREFIX}/lib/libaom.so.3.13.1" )

# Import target "AOM::aom_static" for configuration "Release"
set_property(TARGET AOM::aom_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AOM::aom_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "ASM_NASM;C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaom.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS AOM::aom_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_AOM::aom_static "${_IMPORT_PREFIX}/lib/libaom.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
