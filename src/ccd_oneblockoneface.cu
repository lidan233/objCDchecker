//
// Created by lidan on 11/11/2020.
//

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



__global__ void  gpu_selfcheck(
                                vec3fcu* data,
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

// 如果你的gpu很不错，请使用这个部分
//    __shared__ Lock lo ;
//    if(threadid ==0) {
//        lo.init() ;
//    }


    if(blockid<leftsize && (blockid+leftstart)<size )
    {

        int leftcur = (leftstart + blockid)*3  ;
        vec3fcu leftdata1 = data[ leftcur ] ;
        vec3fcu leftdata2 = data[ leftcur + 1] ;
        vec3fcu leftdata3 = data[ leftcur + 2] ;
        vec3icu leftid = dataid[leftstart+blockid] ;


        for( int i = leftcur+threadid*3+3 ; i < size*3  ; i += threadsize*3 )
        {

            vec3fcu next1 = data[i] ;
            vec3fcu next2 = data[i + 1] ;
            vec3fcu next3 = data[i + 2] ;
            vec3icu nextid = dataid[int(i/3)] ;

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
//                res[*ressize] = vec2fcu(leftstart + blockid,i/3) ;
//                atomicAdd(ressize,1) ;


// 如果你的显卡非常不错 请使用这个部分
//                lo.lock() ;
//                if(mutexx<32)
//                {
//                    contacted[mutexx]=i;
//                }
//                mutexx++;
//                lo.unlock() ;
                int collusion_id = atomicAdd(&mutexx,1) ;
                contacted[collusion_id] = i/3;

            }

        }

        if(threadid<mutexx)
        {
            res[blockid*32+threadid+1] = vec2fcu(leftcur/3,contacted[threadid]) ;
        }
        if(threadid==0)
        {
            res[blockid*32] = vec2fcu(mutexx,leftcur/3) ;
            *ressize += mutexx ;
        }

    }
}




void setting_Data_self_CD(gpu_mesh* cloth,dim3 block,dim3 thread)
{

    std::cout<<" one block for one face"<<std::endl ;
    vec3fcu* tris = cloth->get_alldata() ;
    vec3icu* trisid = cloth->get_dataid() ;
    vec2fcu* result = cloth->init_gpu_result() ;
    int* resultsize = cloth->init_gpu_size() ;


    gpuErrchk( cudaGetLastError() );
    gpu_selfcheck<<<block,thread>>>(tris,
                                    trisid,
                                    result,
                                    resultsize,
                                    0,
                                    cloth->getsize() ,
                                    cloth->getsize()
                                    ) ;


    gpuErrchk( cudaGetLastError() );
    cloth->unget_alldata() ;
    cloth->unget_dataid() ;
}

void getting_Data_self_CD(gpu_mesh* cloth)
{

    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;
    set<int> collusionset ;

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



vec2f* checkSelfCDGPU(gpu_mesh* cloth)
{
    START_GPU
    setting_Data_self_CD(cloth,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_self_CD(cloth) ;
    END_GPU
    return nullptr ;
}


__global__ void gpu_check(vec3fcu* data1,
                          vec3fcu* data2,
                          vec2fcu* res,
                          int* ressize,
                          int leftstart,
                          int leftsize,
                          int rightstart,
                          int rightsize,
                          int allleftsize,
                          int allrightsize)
{
    int blockid = gridDim.x*blockIdx.y + blockIdx.x ;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x ;
    int threadsize = blockDim.x * blockDim.y ;
    __shared__ int mutexx ;
    __shared__ int contacted[32] ;
    mutexx  = 0 ;

    if(blockid<leftsize && (blockid+leftstart)<allleftsize ) {

        int leftcur = (leftstart + blockid) * 3;
        vec3fcu leftdata1 = data1[leftcur];
        vec3fcu leftdata2 = data1[leftcur + 1];
        vec3fcu leftdata3 = data1[leftcur + 2];

        for( int i = rightstart*3 ; i < (rightstart+rightsize)*3 && i< allrightsize*3  ; i += threadsize*3 )
        {
            vec3fcu next1 = data2[i] ;
            vec3fcu next2 = data2[i + 1] ;
            vec3fcu next3 = data2[i + 2] ;


            if(cutri_contact(leftdata1,leftdata2,leftdata3, next1,next2,next3))
            {
                int collusion_id = atomicAdd(&mutexx,1) ;
                contacted[collusion_id] = i;
            }
        }

        if(threadid<mutexx)
        {
            res[blockid*32+threadid+1] = vec2fcu(leftcur/3,contacted[threadid]/3) ;
        }
        if(threadid==0)
        {
            res[blockid*32] = vec2fcu(mutexx,leftcur/3) ;
            *ressize += mutexx ;
        }

    }
}


void setting_Data_CD(gpu_mesh* cloth,gpu_mesh* lion, dim3 block,dim3 thread)
{
    std::cout<<" one block for one face"<<std::endl ;
    vec3fcu* tris_cloth = cloth->get_alldata() ;
    vec3fcu* tris_lion = lion->get_alldata() ;

    vec2fcu* result = cloth->init_gpu_result() ;
    int* resultsize = cloth->init_gpu_size() ;


    gpuErrchk( cudaGetLastError() );
    gpu_check<<<block,thread>>>( tris_cloth,
                                    tris_lion,
                                    result,
                                    resultsize,
                                    0,
                                    cloth->getsize() ,
                                    0,
                                    lion->getsize() ,
                                    cloth->getsize() ,
                                    lion->getsize()
    ) ;

    gpuErrchk( cudaGetLastError() );
    lion->unget_alldata() ;
    cloth->unget_alldata() ;
    cloth->unget_dataid() ;
}

void getting_Data_CD(gpu_mesh* cloth,gpu_mesh* lion)
{
    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;
    set<int> collusion_cloth ;
    set<int> collusion_lion ;


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

vec2f* checkCDGPU(gpu_mesh* cloth,gpu_mesh* lion)
{
    START_GPU
    setting_Data_CD(cloth,lion,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_CD(cloth,lion) ;
    END_GPU
    return nullptr ;
}
