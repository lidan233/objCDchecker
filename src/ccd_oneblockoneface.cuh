//
// Created by lidan on 11/11/2020.
//

#ifndef OBJVIEWER_CCD_ONEBLOCKONEFACE_CUH
#define OBJVIEWER_CCD_ONEBLOCKONEFACE_CUH


#include "vec3fcu.cuh"
#include "gpu_mesh.h"

#include <iostream>

 inline __device__ __host__ int cuproject3(vec3fcu &ax, vec3fcu &p1, vec3fcu &p2, vec3fcu &p3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);

    double mx1 = fmaxmy(P1, P2, P3);
    double mn1 = fminmy(P1, P2, P3);

    if (mn1 > 0) return 0;
    if (0 > mx1) return 0;
    return 1;
}

static inline __device__ __host__ int cuproject6(vec3fcu &ax,
                                 vec3fcu &p1, vec3fcu &p2, vec3fcu &p3,
                                 vec3fcu &q1, vec3fcu &q2, vec3fcu &q3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);
    double Q1 = ax.dot(q1);
    double Q2 = ax.dot(q2);
    double Q3 = ax.dot(q3);

    double mx1 = fmaxmy(P1, P2, P3);
    double mn1 = fminmy(P1, P2, P3);
    double mx2 = fmaxmy(Q1, Q2, Q3);
    double mn2 = fminmy(Q1, Q2, Q3);

    if (mn1 > mx2) return 0;
    if (mn2 > mx1) return 0;
    return 1;
}



static inline bool __device__ __host__ cutri_contact (vec3fcu &P1, vec3fcu &P2, vec3fcu &P3, vec3fcu &Q1, vec3fcu &Q2, vec3fcu &Q3)
{
    vec3fcu p1;
    vec3fcu p2 = P2-P1;
    vec3fcu p3 = P3-P1;
    vec3fcu q1 = Q1-P1;
    vec3fcu q2 = Q2-P1;
    vec3fcu q3 = Q3-P1;

    vec3fcu e1 = p2-p1;
    vec3fcu e2 = p3-p2;
    vec3fcu e3 = p1-p3;

    vec3fcu f1 = q2-q1;
    vec3fcu f2 = q3-q2;
    vec3fcu f3 = q1-q3;

    vec3fcu n1 = e1.cross(e2);
    vec3fcu m1 = f1.cross(f2);


    vec3fcu ef11 = e1.cross(f1);
    vec3fcu ef12 = e1.cross(f2);
    vec3fcu ef13 = e1.cross(f3);
    vec3fcu ef21 = e2.cross(f1);
    vec3fcu ef22 = e2.cross(f2);
    vec3fcu ef23 = e2.cross(f3);
    vec3fcu ef31 = e3.cross(f1);
    vec3fcu ef32 = e3.cross(f2);
    vec3fcu ef33 = e3.cross(f3);



    // now begin the series of tests
    if (!cuproject3(n1, q1, q2, q3)) return false;
    vec3fcu qq1 = -q1 ;
    vec3fcu p21 = p2 - p1 ;
    vec3fcu p31 = p3 - p1 ;
    if (!cuproject3(m1, qq1, p21,p31)) return false;

    if (!cuproject6(ef11, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef12, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef13, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef21, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef22, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef23, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef31, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef32, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(ef33, p1, p2, p3, q1, q2, q3)) return false;

    vec3fcu g1 = e1.cross(n1);
    vec3fcu g2 = e2.cross(n1);
    vec3fcu g3 = e3.cross(n1);


    if (!cuproject6(g1, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(g2, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(g2, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(g3, p1, p2, p3, q1, q2, q3)) return false;

    vec3fcu h1 = f1.cross(m1);
    vec3fcu h2 = f2.cross(m1);
    vec3fcu h3 = f3.cross(m1);


    if (!cuproject6(h1, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(h2, p1, p2, p3, q1, q2, q3)) return false;
    if (!cuproject6(h3, p1, p2, p3, q1, q2, q3)) return false;

    return true;
}


vec2f* checkSelfCDGPU(gpu_mesh* cloth) ;
vec2f* checkCDGPU(gpu_mesh* cloth,gpu_mesh* lion) ;

vec2f* checkSelfCDGPU_no_cpu_memory(gpu_mesh* cloth) ;
vec2f* checkCDGPU_no_cpu_memory(gpu_mesh* cloth,gpu_mesh* lion) ;


vec2f* checkSelfCDGPU_bvh(gpu_mesh* cloth) ;




#endif //OBJVIEWER_CCD_ONEBLOCKONEFACE_CUH
