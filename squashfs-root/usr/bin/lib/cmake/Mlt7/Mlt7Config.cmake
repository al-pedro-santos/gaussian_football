
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was MltConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

set(_supported_components "avformat;decklink;frei0r;jackrack;kdenlive;movit;normalize;oldfilm;opencv;plus;plusgpl;qt6;glaxnimate-qt6;resample;rtaudio;rubberband;sdl2;sox;spatialaudio;vidstab;vorbis;xine;xml")

foreach(_comp ${Mlt7_FIND_COMPONENTS})
  if (NOT _comp IN_LIST _supported_components)
    set(Mlt7_FOUND False)
    set(Mlt7_NOT_FOUND_MESSAGE "Unsupported component: ${_comp}")
  endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/Mlt7Targets.cmake")
