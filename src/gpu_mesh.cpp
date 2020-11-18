//
// Created by lidan on 11/11/2020.
//

#include "gpu_mesh.h"


gpu_mesh::gpu_mesh(mesh *m) {
    data = m ;
    allfaces =(vec3fcu*) malloc(sizeof(vec3fcu)*3*data->getNbFaces()) ;
    allfaces_id = (vec3icu*) malloc(sizeof(vec3icu)*data->getNbFaces()) ;
    allvertexs = (vec3fcu*) malloc(sizeof(vec3fcu)*data->getNbVertices()) ;

    face_size = data->getNbFaces() ;
    vertex_size = data->getNbVertices() ;

    for(int i = 0 ; i < vertex_size; i++)
    {
        allvertexs[i] = data->_vtxs[i] ;
    }


    for (int i=0; i<data->getNbFaces(); i++)
    {
        tri3f &a = data->_tris[i];
        allfaces[i*3] = data->_vtxs[a.id0()] ;
        allfaces[i*3+1] = data->_vtxs[a.id1()] ;
        allfaces[i*3+2] = data->_vtxs[a.id2()] ;
//        std::cout<<allfaces[i*3] << " "<<allfaces[i*3+1]<< " "<<allfaces[i*3+2] <<std::endl ;
        allfaces_id[i] = vec3icu(a.id0(),a.id1(),a.id2()) ;
        if(i<50) std::cout<<"theface "<<i<<" "<<a.id0() << " "<<a.id1()<< " "<<a.id2() <<std::endl ;
    }


#define _USING_BVH_

#ifdef _USING_BVH_
    std::cout<<"begin build bvh"<<std::endl ;
    bvh = getbvhdata(data) ;
    cpu_bvh = bvh->getArrayBoxforGPu() ;
    bvhsize = pow(2,bvh->treearraysize) ;
    cpu_bvh_bound_box = new vec3fcu[bvhsize*2] ;
    cpu_boundbox_id = new int[bvhsize] ;
    for(int i = 0 ; i< bvhsize;i++)
    {
        cpu_bvh_bound_box[i*2] = cpu_bvh[i].box.pmin ;
        cpu_bvh_bound_box[i*2+1] = cpu_bvh[i].box.pmax ;
        cpu_boundbox_id[i] = cpu_bvh[i].id;
    }

    std::cout<<" end build bvh"<<std::endl ;
//    free(cpu_bvh) ;

#endif

}

vec3fcu* gpu_mesh::get_alldata()
{
    gpuErrchk(cudaMalloc((void**) &gpu_allfaces, data->getNbFaces()*3*sizeof(vec3fcu)))  ;
    gpuErrchk(cudaMemcpy(gpu_allfaces, allfaces,data->getNbFaces()*3*sizeof(vec3fcu),cudaMemcpyHostToDevice)) ;
    return gpu_allfaces ;
}



void gpu_mesh::unget_alldata()
{
    gpuErrchk(cudaFree(gpu_allfaces)) ;
    free(allfaces) ;
}


vec3fcu* gpu_mesh::get_vtxsdata()
{
    gpuErrchk(cudaMalloc((void**)&gpu_allvertexs,sizeof(vec3fcu)*vertex_size)) ;
    gpuErrchk(cudaMemcpy(gpu_allvertexs,allvertexs,sizeof(vec3fcu)*vertex_size,cudaMemcpyHostToDevice)) ;
    return gpu_allvertexs ;
}

void gpu_mesh::unget_vtxsdata()
{
    gpuErrchk(cudaFree(gpu_allvertexs)) ;
    free(allvertexs) ;

}



vec3icu* gpu_mesh::get_dataid()
{
    gpuErrchk(cudaMalloc((void**)&gpu_allfaces_id,data->getNbFaces()*sizeof(vec3icu))) ;
    gpuErrchk(cudaMemcpy(gpu_allfaces_id,allfaces_id,data->getNbFaces()*sizeof(vec3icu),cudaMemcpyHostToDevice)) ;
    return gpu_allfaces_id ;
}

void gpu_mesh::unget_dataid()
{
    gpuErrchk(cudaFree(gpu_allfaces_id)) ;
    free(allfaces_id) ;
}

vec2fcu* gpu_mesh::init_gpu_result()
{

    result = (vec2f*)malloc(data->getNbFaces()*sizeof(vec2f)*32) ;
    gpuErrchk(cudaMalloc((void**)&gpu_result,data->getNbFaces()*sizeof(vec2fcu)*32)) ;
    gpuErrchk(cudaMemset(gpu_result,0.0,sizeof(vec2fcu)*data->getNbFaces()*32)) ;
    return gpu_result ;

}

vec2f* gpu_mesh::get_cpu_result()
{
    cudaMemcpy(result,gpu_result,data->getNbFaces()*sizeof(vec2fcu)*32,cudaMemcpyDeviceToHost) ;
    return result ;
}

void gpu_mesh::unget_result()
{
    free(result) ;
    gpuErrchk(cudaFree(gpu_result)) ;
}

int* gpu_mesh::init_gpu_size()
{
    result_size = new int ;
    gpuErrchk(cudaMalloc((void**)&gpu_result_size,sizeof(int))) ;
    gpuErrchk(cudaMemset(gpu_result_size,0,sizeof(int))) ;
    return gpu_result_size ;
}

int* gpu_mesh::get_cpu_size()
{
    gpuErrchk(cudaMemcpy(result_size,gpu_result_size,sizeof(int),cudaMemcpyDeviceToHost)) ;
    return result_size ;
}

void gpu_mesh::unget_size()
{
    free(result_size) ;
    gpuErrchk(cudaFree(gpu_result_size)) ;
}

TreeBoundBox* gpu_mesh::init_bvh() {
#ifdef _USING_BVH_
    checkCudaErrors(cudaMalloc((void**)&gpu_bvh,sizeof(TreeBoundBox)*bvhsize)) ;
    gpuErrchk(cudaMemcpy(gpu_bvh,cpu_bvh,sizeof(TreeBoundBox)*bvhsize,cudaMemcpyHostToDevice))
    return gpu_bvh ;
#else
    return null ;
#endif
}

void gpu_mesh::unget_bvh() {
#ifdef _USING_BVH_
    gpuErrchk(cudaFree(gpu_bvh)) ;
    free(cpu_bvh) ;
#else
    return null ;
#endif
}


vec3fcu* gpu_mesh::init_bvh_box()
{
#ifdef _USING_BVH_
    checkCudaErrors(cudaMalloc((void**)&gpu_bvh_bound_box,sizeof(vec3fcu)*bvhsize*2)) ;
    gpuErrchk(cudaMemcpy(gpu_bvh_bound_box,cpu_bvh_bound_box,sizeof(vec3fcu)*bvhsize*2,cudaMemcpyHostToDevice))

    std::cout<<"successful "<<std::endl ;
    return gpu_bvh_bound_box ;
#else
    return null ;
#endif
}
void gpu_mesh::unget_bvh_box()
{
#ifdef _USING_BVH_
    gpuErrchk(cudaFree(gpu_bvh_bound_box) );
    free(cpu_bvh_bound_box) ;
#else
    return null ;
#endif
}


int* gpu_mesh::init_bvh_id()
{
#ifdef _USING_BVH_
    checkCudaErrors(cudaMalloc((void**)&gpu_boundbox_id,sizeof(int)*bvhsize)) ;
    gpuErrchk(cudaMemcpy(gpu_boundbox_id,cpu_boundbox_id,sizeof(int)*bvhsize,cudaMemcpyHostToDevice)) ;
    std::cout<<"successful "<<std::endl ;
    return gpu_boundbox_id ;
#else
    return null ;
#endif


}


void gpu_mesh::unget_bvhid()
{
#ifdef _USING_BVH_
    gpuErrchk(cudaFree(gpu_boundbox_id) );
    free(cpu_boundbox_id) ;
#else
    return  ;
#endif
}


//int* gpu_mesh::init_bvhsize()
//{
//
//}
//
//void gpu_mesh:: unget_bvhsize()
//{
//
//}


