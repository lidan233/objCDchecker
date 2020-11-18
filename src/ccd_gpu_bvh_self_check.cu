//
// Created by lidan on 12/11/2020.
//

#include "ccd_gpu_bvh_self_check.cuh"
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
    stack[tid*20+stacksize-1] = 0 ;
    int resultsize = 0 ;


    while(stacksize>0)
    {
        int cur = stack[tid*20+stacksize-1] ;
        stacksize-- ;
        TreeBoundBox* t = nodes + cur ;

        if(t->box.interact(box))
        {
            if(t->id >= 0 && t->id!=faceid && resultsize<8)// is leaf
            {
                result[tid*8+resultsize] = t->id ;
                resultsize++ ;
            }else{
                if(stacksize<20 &&cur*2+1<bvhsize&&(nodes+2*cur+1)->id!=-2)  stack[tid*20+stacksize++] = 2*cur+1 ;
                if(stacksize<20 &&cur*2+2<bvhsize&&(nodes+2*cur+2)->id!=-2)  stack[tid*20+stacksize++] = 2*cur+2 ;
            }
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
    __shared__ int mutexx ;
    __shared__ int stack[20*32] ;
    __shared__ int result[8*32] ;
    mutexx  = 0 ;
//    contacted[threadid] = 0 ;
// 如果你的gpu很不错，请使用这个部分
//    __shared__ Lock lo ;
//    if(threadid ==0) {
//        lo.init() ;
//    }
    int cur = ( blockid*threadsize+threadid)  ;
    if(cur<size)
    {

        // 每个线程都有一个自己的Boundbox
        int leftcur = cur*3  ;
        vec3fcu leftdata1 = data[ leftcur ] ;
        vec3fcu leftdata2 = data[ leftcur + 1] ;
        vec3fcu leftdata3 = data[ leftcur + 2] ;
        vec3icu leftid = dataid[cur] ;


        BoundBox box = BoundBox(leftdata1,leftdata2,leftdata3) ;
        //并将其带入到bvh 得到相撞的size
        int mysize = isInteracted(bvhs,box,cur,bvhsize,stack,result,threadid) ;
        for( int i = 0; i < mysize  ; i++ )
        {

            vec3fcu next1 = data[(result[threadid*8+i])*3] ;
            vec3fcu next2 = data[(result[threadid*8+i])*3 + 1] ;
            vec3fcu next3 = data[(result[threadid*8+i])*3 + 2] ;
            vec3icu nextid = dataid[(result[threadid*8+i])] ;

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

//                memory[cur][contacted[threadid]] =
//                contacted[threadid] = i+1 ;
                atomicAdd(&mutexx,1) ;

            }
//            __syncthreads() ;

        }

//        if(threadid<32 && contacted[threadid]!=0)
//        {
//            res[blockid*32+threadid] = vec2fcu(leftcur,contacted[threadid]-1) ;
//        }

        if(threadid==0)
        {
            atomicAdd(ressize,mutexx) ;
//            *ressize += mutexx ;
        }

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
    gpu_self_check_using_trifs<<<block,thread>>>(tris,
                                    trisid,
                                    result,
                                    resultsize,
                                    bvh,
                                    bvhsize,
                                    cloth->getsize()
    ) ;

    gpuErrchk( cudaGetLastError() );
    cloth->unget_alldata() ;
    cloth->unget_bvh() ;
    cloth->unget_dataid() ;
}

void getting_Data_self_CD_bvh(gpu_mesh* cloth)
{
    vec2f* result = cloth->get_cpu_result() ;
    int* resultsize = cloth->get_cpu_size() ;

    std::cout<<"all collusion size is "<<(*resultsize) <<std::endl ;

    cloth->unget_result() ;
    cloth->unget_size() ;

}



vec2f* checkSelfCDGPU_bvh(gpu_mesh* cloth)
{
    START_GPU
    setting_Data_self_CD_bvh(cloth,dim3(1000,1000,1),dim3(32,1,1)) ;
    getting_Data_self_CD_bvh(cloth) ;
    END_GPU
    return nullptr ;
}






