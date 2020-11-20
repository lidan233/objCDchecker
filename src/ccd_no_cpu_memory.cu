//
// Created by lidan on 12/11/2020.
//

#include "ccd_no_cpu_memory.cuh"
#include "ccd_oneblockoneface.cuh"


struct Lock{
    int mutex;
    __device__ Lock(){};
    __device__ void init(){mutex=0;}
    __device__ void lock(){
        while(atomicCAS(&mutex,0,1)!=0);
    }
    __device__ void unlock(){
        atomicExch(&mutex,0);
    }
};



/***
 *
 * @param vtxs  all vertexs in mesh.
 * @param dataid all index of all triangles in mesh
 * @param res result needed to write
 * @param ressize ressize of all data
 * @param leftstart id of face, beginning to check CD
 * @param leftsize face size , begginning to check
 * @param size face size of all
 */
__global__ void  gpu_self_check_using_trifs(vec3fcu* vtxs,
                                            vec3icu* dataid,
                                            vec2fcu* res,
                                            int* ressize,
                                            int leftstart,
                                            int leftsize,
                                            int size)
{
    int blockid = gridDim.x*blockIdx.y + blockIdx.x ;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x ;
    int threadsize = blockDim.x * blockDim.y ;
    __shared__ int mutexx ;
    __shared__ int contacted[32] ;
    mutexx  = 0 ;

    // 如果你的显卡非常好的话
//    __shared__ Lock lo ;
//    if(threadid==0){
//        lo.init() ;
//    }

    if(blockid<leftsize && (blockid+leftstart)<size )
    {

        int leftcur = leftstart + blockid ;
        vec3fcu leftdata1 = vtxs[ dataid[leftcur].x ] ;
        vec3fcu leftdata2 = vtxs[ dataid[leftcur].y ] ;
        vec3fcu leftdata3 = vtxs[ dataid[leftcur].z ] ;
        vec3icu leftid = dataid[leftcur] ;


        for( int i = leftcur+threadid+1 ; i < size  ; i += threadsize )
        {

            vec3fcu next1 = vtxs[ dataid[i].x ] ;
            vec3fcu next2 = vtxs[ dataid[i].y ] ;
            vec3fcu next3 = vtxs[ dataid[i].z ] ;
            vec3icu nextid = dataid[i] ;

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
//                如果你的显卡非常不错 请使用这个部分 但是我再3090上 该操作依然不能work
//                lo.lock() ;
//                if(mutexx<32)
//                {
//                    contacted[mutexx]=i;
//                }
//                mutexx++;
//                lo.unlock() ;

//              记录所有的碰撞操作
                int collusion_id = atomicAdd(&mutexx,1) ;
                contacted[collusion_id] = i;

            }

        }


        if(threadid<mutexx)
        {
            res[blockid*32+threadid+1] = vec2fcu(leftcur,contacted[threadid]) ;
        }
        if(threadid==0)
        {
            res[blockid*32] = vec2fcu(mutexx,leftcur) ;
            *ressize += mutexx ;
        }
    }
}


void setting_Data_self_CD_no_cpu_memo(gpu_mesh* cloth,dim3 block,dim3 thread)
{
    vec3fcu* vertexs = cloth->get_vtxsdata()  ;
    vec3icu* trisid = cloth->get_dataid() ;
    vec2fcu* result = cloth->init_gpu_result() ;
    int* resultsize = cloth->init_gpu_size() ;


    gpuErrchk( cudaGetLastError() );
    START_GPU
    gpu_self_check_using_trifs<<<block,thread>>>(vertexs,
                                    trisid,
                                    result,
                                    resultsize,
                                    0,
                                    cloth->getsize() ,
                                    cloth->getsize()
    ) ;
    END_GPU

    gpuErrchk( cudaGetLastError() );
    cloth->unget_vtxsdata() ;
    cloth->unget_dataid() ;
}

void getting_Data_self_CD_no_cpu_memory(set<int>& collusionset,gpu_mesh* cloth)
{
    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;

    for(int i = 0 ; i < cloth->getsize()*32;i+=32)
    {
        for(int j = 0 ; j < result[i][0]; j++ )
        {
            std::cout<<"gpu check collusion begin between ("<<result[i+j+1][0]<<","<<result[i+j+1][1]<<")"<<std::endl ;
            collusionset.insert(result[i+j+1][0]) ;
            collusionset.insert(result[i+j+1][1]) ;
        }
    }
    std::cout<<"gpu check all collusion size is "<<(*resultsize) <<" " <<collusionset.size()<<std::endl ;

    cloth->unget_result() ;
    cloth->unget_size() ;
}



vec2f* checkSelfCDGPU_no_cpu_memory(set<int>& clothset,gpu_mesh* cloth)
{
    START_GPU
    setting_Data_self_CD_no_cpu_memo(cloth,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_self_CD_no_cpu_memory(clothset,cloth) ;
    END_GPU
    return nullptr ;
}


__global__ void  gpu_check_using_trifs(     vec3fcu* left_vtxs,
                                            vec3icu* left_dataid,
                                            vec3fcu* right_vtxs ,
                                            vec3icu* right_dataid,

                                            vec2fcu* res,
                                            int* ressize,

                                            int leftstart,
                                            int leftsize,
                                            int all_left_size,

                                            int rightstart,
                                            int rightsize,
                                            int all_right_size)
{
    int blockid = gridDim.x*blockIdx.y + blockIdx.x ;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x ;
    int threadsize = blockDim.x * blockDim.y ;
    __shared__ int mutexx ;
    __shared__ int contacted[32] ;
    mutexx  = 0 ;

    // 如果你的显卡非常好的话
//    __shared__ Lock lo ;
//    if(threadid==0){
//        lo.init() ;
//    }

    if(blockid<leftsize && (blockid+leftstart)<all_left_size )
    {

        int leftcur = leftstart + blockid ;
        vec3fcu leftdata1 = left_vtxs[ left_dataid[leftcur].x ] ;
        vec3fcu leftdata2 = left_vtxs[ left_dataid[leftcur].y ] ;
        vec3fcu leftdata3 = left_vtxs[ left_dataid[leftcur].z ] ;


        for( int i = rightstart+threadid ; i < all_right_size  ; i += threadsize )
        {

            vec3fcu next1 = right_vtxs[ right_dataid[i].x ] ;
            vec3fcu next2 = right_vtxs[ right_dataid[i].y ] ;
            vec3fcu next3 = right_vtxs[ right_dataid[i].z ] ;


            if(cutri_contact(leftdata1,leftdata2,leftdata3, next1,next2,next3))
            {
//                如果你的显卡非常不错 请使用这个部分 但是我再3090上 该操作依然不能work
//                lo.lock() ;
//                if(mutexx<32)
//                {
//                    contacted[mutexx]=i;
//                }
//                mutexx++;
//                lo.unlock() ;

//              记录所有的碰撞操作
                int collusion_id = atomicAdd(&mutexx,1) ;
                contacted[collusion_id] = i ;

            }

        }


        if(threadid<mutexx && mutexx<31)
        {
            res[blockid*32+threadid+1] = vec2fcu(leftcur,contacted[threadid]) ;
        }
        if(threadid==0)
        {
            res[blockid*32] = vec2fcu(mutexx,leftcur) ;
            *ressize += mutexx ;
        }

    }
}



void setting_Data_CD_no_cpu_memo(gpu_mesh* cloth,gpu_mesh* lion,dim3 block,dim3 thread)
{
    vec3fcu* cloth_vertexs = cloth->get_vtxsdata()  ;
    vec3icu* cloth_trisid = cloth->get_dataid() ;
    vec2fcu* result = cloth->init_gpu_result() ;
    int* resultsize = cloth->init_gpu_size() ;

    vec3fcu* lion_vertexs = lion->get_vtxsdata() ;
    vec3icu* lion_trisid = lion->get_dataid() ;


    gpu_check_using_trifs<<<block,thread>>>(cloth_vertexs,
                                            cloth_trisid,
                                            lion_vertexs,
                                            lion_trisid,

                                            result,
                                            resultsize,

                                            0,
                                            cloth->getsize(),
                                            cloth->getsize(),

                                            0,
                                            lion->getsize(),
                                            lion->getsize()
                                            ) ;


    gpuErrchk( cudaGetLastError() );
    cloth->unget_vtxsdata() ;
    cloth->unget_dataid() ;
    lion->unget_vtxsdata() ;
    lion->unget_dataid() ;
}


void getting_Data_CD_no_cpu_memory(set<int>& collusion_cloth,set<int>& collusion_lion,gpu_mesh* cloth,gpu_mesh* lion)
{
    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;

    for(int i = 0 ; i < cloth->getsize()*32;i+=32)
    {
        for(int j = 0 ; j < result[i][0]; j++ )
        {
            std::cout<<"gpu check collusion begin between ("<<result[i+j+1][0]<<","<<result[i+j+1][1]<<")"<<std::endl ;
            collusion_cloth.insert(result[i+j+1][0]) ;
            collusion_lion.insert(result[i+j+1][1]) ;
        }
    }
    std::cout<<"gpu check all collusion size is "<<(*resultsize) <<std::endl ;
    std::cout<<"cloth collusion size is "<<collusion_cloth.size()<<std::endl ;
    std::cout<<"lion collusion size is "<< collusion_lion.size() <<std::endl ;

    cloth->unget_result() ;
    cloth->unget_size() ;
}



vec2f* checkCDGPU_no_cpu_memory(set<int>& clothset,set<int>& clionset,gpu_mesh* cloth,gpu_mesh* lion)
{
    START_GPU
    setting_Data_CD_no_cpu_memo(cloth,lion,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_CD_no_cpu_memory(clothset,clionset,cloth,lion) ;
    END_GPU
    return nullptr ;
}
