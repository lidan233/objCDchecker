//
// Created by lidan on 13/11/2020.
//

#ifndef OBJVIEWER_UTIL_CUH
#define OBJVIEWER_UTIL_CUH


#include "vec3fcu.cuh"
#include "cmesh.h"

#include <map>
#include <iostream>
#include <cuda_runtime.h>

struct BoundBox{
public:
    vec3fcu pmin ;
    vec3fcu pmax ;
    __host__ __device__ BoundBox()
    {
        pmin = vec3fcu(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max()) ;

        pmax = vec3fcu(std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::min()) ;
    }
    __host__ __device__ BoundBox(vec3fcu a)
    {
        pmin = a ;
        pmax = a ;
    }
    __host__ __device__ BoundBox(vec3fcu a,vec3fcu b,vec3fcu c)
    {
        vec3fcu max = a.vmax(b) ;
        pmax = max.vmax(c) ;

        vec3fcu min = a.vmin(b) ;
        pmin = min.vmin(c) ;
    }

    void __host__ __device__ set(BoundBox ano)
    {
        pmin = ano.pmin ;
        pmax = ano.pmax ;
    }

    void __host__ __device__ merge(BoundBox ano)
    {
        pmin = pmin.vmin(ano.pmin) ;
        pmax = pmax.vmax(ano.pmax) ;
    }

    __host__ __device__ void  merge(vec3fcu d)
    {
        pmin = pmin.vmin(d) ;
        pmax = pmax.vmax(d) ;
    }

    int __host__ __device__  MaximumExtent()
    {
        vec3fcu d = vec3fcu(pmax.x - pmin.x, pmax.y - pmin.y,pmax.z - pmin.z); // diagonal
        if (d.x > d.y && d.x > d.z) {
            return 0;
        }
        if (d.y > d.z) {
            return 1;
        }
        else {
            return 2;
        }
    }

    bool __host__ __device__ interact(BoundBox& e) const
    {
        vec3fcu c = e.getmidP() - this->getmidP() ;
        c.abs() ;
        vec3fcu l = ((pmax-pmin) + (e.pmax-e.pmin))/2 ;
        return c<l;
    }

    bool __host__  interacted(BoundBox& e) const
    {
        vec3fcu c = e.getmidP() - this->getmidP() ;
        c.abs() ;
        vec3fcu l = ((pmax-pmin) + (e.pmax-e.pmin))/2 ;
        return c<l;
    }

    vec3fcu __host__ __device__ getmidP() const
    {
        return vec3fcu(pmin.x*0.5+pmax.x*0.5,
                       pmin.y*0.5+pmax.y*0.5,
                       pmin.z*0.5+pmax.z*0.5);
    }
    bool __host__ __device__ operator==(const BoundBox &another) const
    {
        return (pmin ==another.pmin) && (pmax == another.pmax) ;
    }
};


struct BVHInfo{
    int id_ ;
    BoundBox box ;
    vec3fcu centroid;

    BVHInfo() = default;
    BVHInfo(const BoundBox& box,int id):
            id_(id),
            box(box),
            centroid(box.getmidP()){}
};

enum axis{
    x,y,z
};


class BVHNode{
public:
    BoundBox node_box ;
    BVHNode* children[2] ;
    int split_axis,nid,number_ofinfo ;

    void leafNode(int id,int  n,BoundBox& box)
    {
        node_box = box ;
        nid = id ;
        box = node_box ;
        number_ofinfo = n;
        children[0] = nullptr ;
        children[1] = nullptr ;
    }

    void InNode(int split,BVHNode* left,BVHNode* right )
    {
        children[0] = left ;
        children[1] = right ;
        split_axis = split ;
        number_ofinfo = 0 ;
        nid = -1 ;
        node_box = BoundBox() ;
        node_box.merge(left->node_box) ;
        node_box.merge(right->node_box) ;
    }

};

class ArrayNode{
    BoundBox *box ;
    union{
        int leftoffset ;
        int rightoffset ;
    };
    int number_ofinfo ;
    int axis ;
};

struct TreeBoundBox{
    BoundBox box ;
    int id = -2 ;
};


class BVH{
public:
    BVHNode* root;
    TreeBoundBox* result = nullptr ;
    int treearraysize = 0 ;
    std::map<int,BVHNode*> maps ;

    BVH(BVHInfo* bvhInfos,int size ) ;
    BVHNode* SplitBuild(BVHInfo* infos,int begin,int end, int *allnodes) ;
    ArrayNode* getArrayBVH_no_redundancy(BVHNode* root)=delete ;
    int getDeep(BVHNode* root,int deep) ;
    int getsize(BVHNode* root) ;
    BoundBox* getArrayBVH(BVHNode* root) =delete;
    void getArrayBox(BVHNode* root,int offset) ;
    TreeBoundBox* getArrayBoxforGPu() { return result;}
    BVHNode* getBVHTree() { return root;}
};


static inline BVH* getbvhdata(mesh* data)
{

    int face_size = data->getNbFaces() ;
    BVHInfo* bvhs = new BVHInfo[face_size] ;
    for(int i =0 ; i< data->getNbFaces();i++ )
    {
        vec3fcu r1 = data->_vtxs[data->_tris[i].id0()] ;
        vec3fcu r2 = data->_vtxs[data->_tris[i].id1()] ;
        vec3fcu r3 = data->_vtxs[data->_tris[i].id2()] ;


        BoundBox t = BoundBox(r1,r2,r3) ;
        bvhs[i] = BVHInfo(t,i) ;
        if(i<50) {
            std::cout<<"the "<<i<<" face is:"<<data->_tris[i].id0()<<" "<<data->_tris[i].id1()<<" "<<data->_tris[i].id2()<<std::endl ;
            std::cout<<"the BoundBox pmin is "<<bvhs[i].box.pmin[0]<<" "<<bvhs[i].box.pmin[1]<<" "<<bvhs[i].box.pmin[2]<<std::endl ;
            std::cout<<"the BoundBox pmin is "<<bvhs[i].box.pmax[0]<<" "<<bvhs[i].box.pmax[1]<<" "<<bvhs[i].box.pmax[2]<<std::endl ;
        }

    }

//    * bvh_result = new BVH(bvhs,face_size) ;
    BVH* bvh_result = new BVH(bvhs,face_size) ;
    int count = 0 ;
    for(int i = 0 ; i<pow(2,bvh_result->treearraysize);i++)
    {
        if(bvh_result->result[i].id>=0) count++ ;
    }
//    std::cout<<count<<std::endl ;

    return bvh_result ;
}

#endif //OBJVIEWER_UTIL_CUH
