//
// Created by lidan on 17/11/2020.
//

#ifndef OBJVIEWER_CCD_CPU_BVH_CD_H
#define OBJVIEWER_CCD_CPU_BVH_CD_H
#include "vec3fcu.cuh"
#include "ccd_oneblockoneface.cuh"

#include "vec3f.h"
#include "util.cuh"
#include "gpu_mesh.h"

#include <vector>
#include <set>

class ccd_cpu_bvh_cd {
private:
    std::set<int> collusion ;

public:
    ccd_cpu_bvh_cd() ;
    void init() ;
    std::set<int>* checkSelfCDByBVH(gpu_mesh* mesh1) ;
    std::set<int>* checkCDByBVH(gpu_mesh* mesh1,gpu_mesh* mesh2) ;


};


#endif //OBJVIEWER_CCD_CPU_BVH_CD_H
