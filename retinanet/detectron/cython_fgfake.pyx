# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

@cython.boundscheck(False)
def find_fg_fake_inds(
        np.ndarray[DTYPE_t, ndim=1] enable_inds,
        np.ndarray[DTYPE_t, ndim=1] fg_inds,
        DTYPE_t fg_value):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int fake_num = 0
    cdef unsigned int B = enable_inds.shape[0]
    cdef unsigned int F = fg_inds.shape[0]
    cdef unsigned int b
    cdef unsigned int f
    with nogil:
        for b in range(B):
            for f in range(F):
                if enable_inds[b] == fg_inds[f]:
                    fake_num += 1
                     
    cdef np.ndarray[DTYPE_t, ndim=1] fg_fake_inds = np.ones((fake_num, ), dtype=DTYPE)
    cdef unsigned int fa
    with nogil:
        for fa in range(fake_num):
            fg_fake_inds[fa] = fg_value
    return fg_fake_inds
