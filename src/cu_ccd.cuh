//
// Created by lidan on 12/11/2020.
//

#ifndef OBJVIEWER_CU_CCD_CUH
#define OBJVIEWER_CU_CCD_CUH



#include<helper_cuda.h>
#include"cmesh.h"
#include <vector>
#include<iostream>

//#include<cuda_runtime.h>
//#include<thrust/host_vector.h>
//#include<thrust/device_vector.h>
#define HANDLE_ERROR checkCudaErrors

#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}

#define START_CPU {\
double start = omp_get_wtime();

#define END_CPU \
double end = omp_get_wtime();\
double duration = end - start;\
printf("CPU Time used: %3.1f ms\n", duration * 1000);}

class CollisionDetect{
    mesh *obj_a_,*obj_b_;
    set<int>& a_set_;
    set<int>& b_set_;
    //顶点数目
    unsigned int  obj_a_tri_num_,obj_b_tri_num_;
    //三角形数目
    unsigned int obj_a_vtx_num_,obj_b_vtx_num_;
    //host
    tri3f *obj_a_tris_,*obj_b_tris_;
    vec3f *obj_a_vtxs_,*obj_b_vtxs_;
    vec3f *obj_a_data_,*obj_b_data_;
    //device
    tri3f *d_obj_a_tris_,*d_obj_b_tris_;
    vec3f *d_obj_a_vtxs_,*d_obj_b_vtxs_;
    vec3f *d_obj_a_data_,*d_obj_b_data_;

    //result
    unsigned int * d_res;
    size_t pitch;
    unsigned int * h_res;
    double* d_test,*h_test;

    const bool is_self_cd;
public:
    explicit CollisionDetect(mesh* obj_a,mesh* obj_b,set<int>& a_set,set<int>& b_set):obj_a_(obj_a),obj_b_(obj_b),a_set_(a_set),b_set_(b_set),is_self_cd(false){
        assert(obj_a_ && obj_b_);
        if(obj_a_->getNbFaces()<obj_b_->getNbFaces()){
            std::swap(obj_a_,obj_b_);
            std::swap(a_set_,b_set_);
        }
        obj_a_tri_num_=obj_a_->getNbFaces();
        obj_a_vtx_num_=obj_a_->getNbVertices();
        obj_b_tri_num_=obj_b_->getNbFaces();
        obj_b_vtx_num_=obj_b_->getNbVertices();
        obj_a_tris_=obj_a_->_tris;
        obj_a_vtxs_=obj_a_->_vtxs;
        obj_b_tris_=obj_b_->_tris;
        obj_b_vtxs_=obj_b_->_vtxs;
        std::cout<<"CollisionDetect create finish..."<<std::endl;
    }
    explicit CollisionDetect(mesh* obj_a,set<int>& a_set):obj_a_(obj_a),a_set_(a_set),b_set_(a_set),is_self_cd(true){
        assert(obj_a);
        obj_a_tri_num_=obj_a_->getNbFaces();
        obj_a_vtx_num_=obj_a_->getNbVertices();
        obj_a_tris_=obj_a_->_tris;
        obj_a_vtxs_=obj_a_->_vtxs;
        std::cout<<"SelfCollisionDetect create finish..."<<std::endl;
    }

    CollisionDetect()=default;
    void SetMeshObj(mesh* obj_a,mesh* obj_b){obj_a_=obj_a;obj_b_=obj_b;}

    void TriContactDetect();
    void TriContactDetect(const tri3f* obj_a_tris,const tri3f* obj_b_tris,const vec3f* obj_a_vtxs,const vec3f* obj_b_vtxs)=delete;
    bool TriContact(const vec3f& P1,const vec3f& P2,const vec3f& P3,const vec3f& Q1,const vec3f& Q2,const vec3f& Q3)=delete;
    bool SurfaceContact(const vec3f& ax,const vec3f& p1,const vec3f& p2,const vec3f& p3)=delete;
    bool RotationContact(const vec3f& ax,const vec3f& p1,const vec3f& p2,const vec3f& p3,const vec3f& q1,const vec3f& q2,const vec3f& q3)=delete;
private:
    void PrepareGPUData();
    void GenerateTrisData();
};




#endif //OBJVIEWER_CU_CCD_CUH
