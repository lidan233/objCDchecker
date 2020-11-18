//
// Created by lidan on 12/11/2020.
//

#ifndef OBJVIEWER_TRI_CU_CUH
#define OBJVIEWER_TRI_CU_CUH

#include "forceline.h"

class cutri3f {
public:
    unsigned int _ids[3];

    FORCEINLINE __host__ __device__ cutri3f() {
        _ids[0] = _ids[1] = _ids[2] = -1;
    }

    FORCEINLINE __host__ __device__ cutri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
        set(id0, id1, id2);
    }

    FORCEINLINE __host__ __device__ void set(unsigned int id0, unsigned int id1, unsigned int id2) {
        _ids[0] = id0;
        _ids[1] = id1;
        _ids[2] = id2;
    }

    FORCEINLINE __host__ __device__  unsigned int id(int i) const { return _ids[i]; }
    FORCEINLINE __host__ __device__  unsigned int id0() const {return _ids[0];}
    FORCEINLINE __host__ __device__  unsigned int id1() const {return _ids[1];}
    FORCEINLINE __host__ __device__  unsigned int id2() const {return _ids[2];}
};



#endif //OBJVIEWER_TRI_CU_CUH
