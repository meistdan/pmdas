#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# CUDA compilation doesn't pick up the mdas include path target property.
#include_directories( include )

cuda_wrap_srcs( mdas OBJ generated_files kdgrid.cu "" OPTIONS "" )

add_library( mdas STATIC
  kdgrid.h
  aabb.h
  buffer.h
  point.h
  kdgrid.cu
  ${generated_files}
)

set_target_properties(mdas PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories( mdas PUBLIC include )

# These CUDA files need to be compiled as OBJ, not as PTX
set_source_files_properties( kdgrid.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

# This file is just for reference in the IDE, don't compile it.
#set_source_files_properties( optixPagingImpl.cpp PROPERTIES HEADER_FILE_ONLY TRUE )

set_property(TARGET mdas PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
