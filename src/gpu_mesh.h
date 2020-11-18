//
// Created by lidan on 11/11/2020.
//

#ifndef OBJVIEWER_GPU_MESH_H
#define OBJVIEWER_GPU_MESH_H

#include "cmesh.h"
#include "vec3fcu.cuh"
#include "util.cuh"

#include <iostream>
#include <omp.h>
#include <map>


#define START_GPU \
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0)); \
                           \


#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));



class gpu_mesh {
private:
    mesh* data ;
    BVH* bvh = nullptr ;
    bool usingBvh = false ;
    TreeBoundBox* cpu_bvh = nullptr ;
    TreeBoundBox* gpu_bvh ;
    int bvhsize = 0 ;

    vec3fcu* cpu_bvh_bound_box ;
    int* cpu_boundbox_id ;
    vec3fcu* gpu_bvh_bound_box ;
    int* gpu_boundbox_id ;

    vec3fcu* allfaces ;
    vec3fcu* gpu_allfaces ;
    int face_size ;

    vec3fcu* allvertexs ;
    vec3fcu* gpu_allvertexs ;
    int vertex_size ;

    vec3icu* allfaces_id ;
    vec3icu* gpu_allfaces_id ;
    int faceidsize ;

    vec2f* result ;
    vec2fcu* gpu_result ;

    int* result_size ;
    int* gpu_result_size ;




public:
    gpu_mesh(mesh* m) ;
    int getsize(){ return data->getNbFaces() ;}

    vec3fcu* cpu_get_alldata(){ return allfaces ; }
    vec3icu* cpu_get_faceid(){return allfaces_id ;}

    vec3fcu* get_alldata() ;
    void unget_alldata() ;

    vec3fcu* get_vtxsdata() ;
    void unget_vtxsdata() ;

    vec3icu* get_dataid() ;
    void unget_dataid() ;

    vec2fcu* init_gpu_result() ;
    vec2f* get_cpu_result() ;
    void unget_result() ;

    int* init_gpu_size() ;
    int* get_cpu_size()  ;
    void unget_size() ;

    TreeBoundBox* init_bvh() ;
    void unget_bvh() ;

    vec3fcu* init_bvh_box() ;
    void unget_bvh_box() ;

    int* init_bvh_id() ;
    void unget_bvhid() ;

    TreeBoundBox* get_cpu_bvh(){ return cpu_bvh ;}
    BVHNode* get_cpu_bvh_tree() { return bvh->root ;}
    std::map<int,BVHNode*> get_cpu_bvh_map() { return bvh->maps;}

    int get_face_size() { return face_size ;}
    int get_vertex_size() { return vertex_size ;}
    int get_bvh_size(){ return bvhsize ;}


//    int* init_bvhsize() ;
//    void unget_bvhsize() ;

};



#endif //OBJVIEWER_GPU_MESH_H