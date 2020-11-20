//
// Created by lidan on 12/11/2020.
//

#include "ccd_gpu_bvh_check.cuh"
#include "vec3fcu.cuh"
#include "ccd_oneblockoneface.cuh"
#include "util.cuh"


//叶子节点的id就是对应的face的id
//非叶子节点的id为-1
//冗余节点的id是-2
// stack size for every tid : 20
// result size for every result : 8
static __device__ __host__ int isInteracted(TreeBoundBox* nodes,
                                            BoundBox box,
                                            int faceid,
                                            int bvhsize,
                                            int* stack,
                                            int* result,
                                            int tid)
{
    if( nodes == nullptr ) return 0;

    int stacksize = 1 ;
    stack[tid*32+stacksize-1] = 0 ;
    int resultsize = 0 ;


    while(stacksize>0)
    {
        int cur = stack[tid*32+stacksize-1] ;
        stacksize-- ;
        TreeBoundBox* t = nodes + cur ;

        if(t->box.interact(box))
        {
            if(t->id >= 0)// is leaf
            {
                if(t->id>faceid)
                {
                    if(resultsize<32) result[tid*32+resultsize] = t->id ;
                    resultsize++ ;
                }
                continue ;
            }
            if(stacksize<32 && (2*cur+1)<bvhsize && (nodes+2*cur+1)->id!=-2)  stack[tid*32+stacksize++] = 2*cur+1 ;
            if(stacksize<32 && (2*cur+1)<bvhsize && (nodes+2*cur+2)->id!=-2)  stack[tid*32+stacksize++] = 2*cur+2 ;
        }
    }
    return resultsize ;
}

__global__ void  gpu_self_check_using_trifs(vec3fcu* data,
                                            vec3icu* dataid,
                                            vec2fcu* res,
                                            int* ressize,
                                            TreeBoundBox* bvhs,
                                            int bvhsize ,
                                            int size)
{
    int blockid = gridDim.x*blockIdx.y + blockIdx.x ;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x ;
    int threadsize = blockDim.x * blockDim.y ;
    __shared__ int mutexx[32] ;
    __shared__ int stack[32*32] ;
    __shared__ int result[32*32] ;
    mutexx[threadid]  = 0 ;


    int cur = ( blockid*threadsize+threadid)  ;
    if(cur<size)
    {

        int leftcur = cur*3  ;
        vec3fcu leftdata1 = data[ leftcur ] ;
        vec3fcu leftdata2 = data[ leftcur + 1] ;
        vec3fcu leftdata3 = data[ leftcur + 2] ;
        vec3icu leftid = dataid[cur] ;


        BoundBox box = BoundBox(leftdata1,leftdata2,leftdata3) ;
        //并将其带入到bvh 得到相撞的size
        int mysize = isInteracted(bvhs,box,cur,bvhsize,stack,result,threadid) ;
        for( int i = 0; i < (mysize>32?32:mysize)  ; i++ )
        {
            vec3fcu next1 = data[(result[threadid*32+i])*3] ;
            vec3fcu next2 = data[(result[threadid*32+i])*3 + 1] ;
            vec3fcu next3 = data[(result[threadid*32+i])*3 + 2] ;
            vec3icu nextid = dataid[(result[threadid*32+i])] ;

            bool cons = true ;
            for(int g = 0 ; g< 3;g++)
            {
                for(int h = 0 ; h <3 ;h++)
                {
                    if(leftid[g]==nextid[h])
                    {
                        cons = false ;
                        break ;
                    }
                }
            }


            if(cons&&cutri_contact(leftdata1,leftdata2,leftdata3, next1,next2,next3))
            {
                mutexx[threadid] = mutexx[threadid]+1 ;
                if(mutexx[threadid]<31) res[cur*32+mutexx[threadid]] = vec2fcu(cur,result[threadid*32+i]) ;
            }
        }
        res[cur*32] = vec2fcu(mutexx[threadid],0) ;
        atomicAdd(ressize,mutexx[threadid]) ;

    }

}


void setting_Data_self_CD_bvh(gpu_mesh* cloth,dim3 block,dim3 thread)
{
    vec3fcu* tris = cloth->get_alldata() ;
    vec3icu* trisid = cloth->get_dataid() ;
    vec2fcu* result = cloth->init_gpu_result() ;
    int* resultsize = cloth->init_gpu_size() ;
//    vec3fcu* bvh_boundbox = cloth->init_bvh_box() ;
    TreeBoundBox* bvh = cloth->init_bvh() ;
//    int* bvhid = cloth->init_bvh_id() ;
    int bvhsize= cloth->get_bvh_size() ;


    gpuErrchk( cudaGetLastError() );
    START_GPU
    gpu_self_check_using_trifs<<<block,thread>>>(tris,
                                    trisid,
                                    result,
                                    resultsize,
                                    bvh,
                                    bvhsize,
                                    cloth->getsize()
    ) ;
    END_GPU

    gpuErrchk( cudaGetLastError() );
    cloth->unget_alldata() ;
    cloth->unget_bvh() ;
    cloth->unget_dataid() ;
}

void getting_Data_self_CD_bvh(set<int>& collusionset, gpu_mesh* cloth)
{
    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;



    std::cout<<"all collusion size is "<<(*resultsize) <<std::endl ;

    for(int i = 0 ; i < cloth->getsize()*32;i+=32)
    {
        if(result[i][0]<=0) continue;
//        std::cout<<i/32<<" "<<result[i][0]<<std::endl ;
        for(int j = 0 ; j < result[i][0];j++)
        {
            std::cout<<"gpu check collusion begin between ("<<result[i+j+1][0]<<","<<result[i+j+1][1]<<")"<<std::endl ;
            collusionset.insert(result[i+j+1][0]) ;
            collusionset.insert(result[i+j+1][1]) ;
        }
    }
    std::cout<<"gpu check all collusion size is "<<(*resultsize) <<std::endl ;
    std::cout<<"cloth collusion size is "<<collusionset.size()<<std::endl ;
    cloth->unget_result() ;
    cloth->unget_size() ;

}



vec2f* checkSelfCDGPU_bvh(set<int>& clothset,gpu_mesh* cloth)
{
    START_GPU
    setting_Data_self_CD_bvh(cloth,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_self_CD_bvh(clothset, cloth) ;
    END_GPU
    return nullptr ;
}


static __device__ __host__ int isInteracted_cd(TreeBoundBox* nodes,
                                            BoundBox box,
                                            int bvhsize,
                                            int* stack,
                                            int* result,
                                            int tid)
{
    if( nodes == nullptr ) return 0;

    int stacksize = 1 ;
    stack[tid*32+stacksize-1] = 0 ;
    int resultsize = 0 ;


    while(stacksize>0)
    {
        int cur = stack[tid*32+stacksize-1] ;
        stacksize-- ;
        TreeBoundBox* t = nodes + cur ;

        if(t->box.interact(box))
        {
            if(t->id >= 0)// is leaf
            {
                if(resultsize<32) result[tid*32+resultsize] = t->id ;
                resultsize++ ;
                continue ;
            }
            if(stacksize<32 && (2*cur+1)<bvhsize && (nodes+2*cur+1)->id!=-2)  stack[tid*32+stacksize++] = 2*cur+1 ;
            if(stacksize<32 && (2*cur+1)<bvhsize && (nodes+2*cur+2)->id!=-2)  stack[tid*32+stacksize++] = 2*cur+2 ;
        }
    }
    return resultsize ;
}



__global__ void  gpu_check_using_trifs(vec3fcu* data1,
                                       vec3fcu* data2,
                                       vec2fcu* res,
                                       int* ressize,
                                       TreeBoundBox* bvhs_data2,
                                       int bvh_size2 ,
                                       int size1,
                                       int size2)
{
    int blockid = gridDim.x*blockIdx.y + blockIdx.x ;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x ;
    int threadsize = blockDim.x * blockDim.y ;
    __shared__ int mutexx[32] ;
    __shared__ int stack[32*32] ;
    __shared__ int result[32*32] ;
    mutexx[threadid]  = 0 ;

    int cur = ( blockid*threadsize+threadid)  ;
    if(cur<size1)
    {

        int leftcur = cur*3  ;
        vec3fcu leftdata1 = data1[ leftcur ] ;
        vec3fcu leftdata2 = data1[ leftcur + 1 ] ;
        vec3fcu leftdata3 = data1[ leftcur + 2 ] ;


        BoundBox box = BoundBox(leftdata1,leftdata2,leftdata3) ;
        int mysize = isInteracted_cd(bvhs_data2,box,bvh_size2,stack,result,threadid) ;
        for( int i = 0; i < (mysize>32?32:mysize)  ; i++ )
        {
            vec3fcu next1 = data2[(result[threadid*32+i])*3] ;
            vec3fcu next2 = data2[(result[threadid*32+i])*3 + 1] ;
            vec3fcu next3 = data2[(result[threadid*32+i])*3 + 2] ;



            if(cutri_contact(leftdata1,leftdata2,leftdata3, next1,next2,next3))
            {
                mutexx[threadid] = mutexx[threadid]+1 ;
                if(mutexx[threadid]<31) res[cur*32+mutexx[threadid]] = vec2fcu(cur,result[threadid*32+i]) ;
            }
        }
        res[cur*32] = vec2fcu(mutexx[threadid],0) ;
        atomicAdd(ressize,mutexx[threadid]) ;

    }

}


void setting_Data_CD_bvh(gpu_mesh* cloth,gpu_mesh* lion,dim3 block,dim3 thread)
{
    vec3fcu* tris_cloth = cloth->get_alldata() ;
    vec3fcu* tris_lion = lion->get_alldata() ;

    vec2fcu* result = cloth->init_gpu_result() ;
    int* resultsize = cloth->init_gpu_size() ;

    TreeBoundBox* bvh = lion->init_bvh() ;
    int bvhsize= lion->get_bvh_size() ;


    gpuErrchk( cudaGetLastError() );
    gpu_check_using_trifs<<<block,thread>>>(tris_cloth,
                                                 tris_lion,
                                                 result,
                                                 resultsize,
                                                 bvh,
                                                 bvhsize,
                                                 cloth->getsize(),
                                                 lion->getsize()
    ) ;

    gpuErrchk( cudaGetLastError() );
    cloth->unget_alldata() ;
    lion->unget_alldata() ;
    lion->unget_bvh() ;

}

void getting_Data_CD_bvh(set<int>& clothset,gpu_mesh* cloth,gpu_mesh* lion)
{
    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;
    set<int> collusioncloth ;
    set<int> collusionlion ;

    std::cout<<"all collusion size is "<<(*resultsize) <<std::endl ;

    for(int i = 0 ; i < cloth->getsize()*32;i+=32)
    {
        if(result[i][0]<=0) continue;
//        std::cout<<i/32<<" "<<result[i][0]<<std::endl ;
        for(int j = 0 ; j < result[i][0];j++)
        {
            std::cout<<"gpu check collusion begin between ("<<result[i+j+1][0]<<","<<result[i+j+1][1]<<")"<<std::endl ;
            collusioncloth.insert(result[i+j+1][0]) ;
            collusionlion.insert(result[i+j+1][1]) ;
        }
    }
    std::cout<<"gpu check all collusion size is "<<(*resultsize) <<std::endl ;
    std::cout<<"cloth collusion size is "<<collusioncloth.size()<<std::endl ;
    std::cout<<"lion collusion size is "<<collusionlion.size()<<std::endl ;
    cloth->unget_result() ;
    cloth->unget_size() ;

}

vec2f* checkCDGPU_bvh(set<int>& clothset,set<int>& clionset,gpu_mesh* cloth,gpu_mesh* lion)
{
    START_GPU
    setting_Data_CD_bvh(cloth,lion,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_CD_bvh(clothset,cloth,lion) ;
    END_GPU
    return nullptr ;
}







