cmake_minimum_required (VERSION 3.5)
project(ARMCL_MWE)

if((CMAKE_CXX_COMPILER_ID MATCHES GNU) OR (CMAKE_CXX_COMPILER_ID MATCHES Clang))
   set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++17")
   set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
   set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -ffast-math")
endif()

list(APPEND MWE_INCLUDE_DIRS PUBLIC ${ARMCL_MWE_SOURCE_DIR}/src/)

list(APPEND MWE_INCLUDE_DIRS PUBLIC ${ORPHEUS_SOURCE_DIR}/include/)

# Find libraries
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
message(${CMAKE_MODULE_PATH})

# ---[ ACL
find_package(ARMCL REQUIRED)
add_definitions(-DARMCL)
# add subdirectories to includes
list(APPEND MWE_INCLUDE_DIRS PUBLIC ${ARMCL_INCLUDE})
list(APPEND MWE_INCLUDE_DIRS PUBLIC ${ARMCL_INCLUDE}/include)
list(APPEND MWE_INCLUDE_DIRS PUBLIC ${ARMCL_INCLUDE}/include/CL)

set(__list ${ARMCL_LIBRARIES})
separate_arguments(__list)
list(REMOVE_DUPLICATES __list)
foreach(i ${__list})
  list(APPEND MWE_LINKER_LIBS PUBLIC ${i})
endforeach()

add_subdirectory(src)



