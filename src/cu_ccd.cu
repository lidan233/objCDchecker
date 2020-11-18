//
// Created by lidan on 12/11/2020.
//

#include "cu_ccd.cuh"
//
// Created by wyz on 20-11-2.
//
#define uint unsigned int

uint32_t blocks_per_grid;
const uint32_t threads_per_block=32;
const uint32_t MAX_CONTACT_NUM=threads_per_block-1;
//using namespace thrust;
void CollisionDetect::PrepareGPUData()
{
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout<<"PrepareGPUData"<<std::endl;

//    std::cout<<sizeof(tri3f)<<" "<<sizeof(vec3f)<<std::endl;
    std::cout<<"obj_a_tri_num_ is: "<<obj_a_tri_num_<<std::endl;
    std::cout<<"obj_a_vtx_num_ is: "<<obj_a_vtx_num_<<std::endl;
    checkCudaErrors(cudaMalloc((void**)&d_obj_a_tris_,obj_a_tri_num_*sizeof(tri3f )));
    checkCudaErrors(cudaMalloc((void**)&d_obj_a_vtxs_,obj_a_vtx_num_*sizeof(vec3f )));
//    std::cout<<obj_a_tri_num_*sizeof(tri3f)<<std::endl;
//    std::cout<<d_obj_a_tris_<<" "<<obj_a_tris_<<std::endl;
    checkCudaErrors(cudaMemcpy(d_obj_a_tris_,obj_a_tris_,obj_a_tri_num_*sizeof(tri3f),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_obj_a_vtxs_,obj_a_vtxs_,obj_a_vtx_num_*sizeof(vec3f),cudaMemcpyHostToDevice));

    if(!is_self_cd){
        //prepare data for obj b
        checkCudaErrors(cudaMalloc((void**)&d_obj_b_tris_,obj_b_tri_num_*sizeof(tri3f)));
        checkCudaErrors(cudaMalloc((void**)&d_obj_b_vtxs_,obj_b_vtx_num_*sizeof(vec3f )));

        checkCudaErrors(cudaMemcpy(d_obj_b_tris_,obj_b_tris_,obj_b_tri_num_*sizeof(tri3f),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_obj_b_vtxs_,obj_b_vtxs_,obj_b_vtx_num_*sizeof(vec3f),cudaMemcpyHostToDevice));
    }

    std::cout<<"input data alloc finish..."<<std::endl;

    //   checkCudaErrors(cudaMallocPitch((void**)&d_res,&pitch,sizeof(uint)*1024,obj_a_tri_num_));
    checkCudaErrors(cudaMalloc((void**)&d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block));
    pitch=32;
    std::cout<<"pitch is: "<<pitch<<std::endl;
    h_res=(uint*)malloc(obj_a_tri_num_*sizeof(uint)*threads_per_block);

//    checkCudaErrors(cudaMalloc((void**)&d_test,sizeof(double)*9));
//    h_test=(double*)malloc(9*sizeof(double));

    std::cout<<"data prepare finish..."<<std::endl;
}
void CollisionDetect::GenerateTrisData()
{
    std::cout<<"GenerateTrisData"<<std::endl;
    obj_a_data_=(vec3f*)malloc(obj_a_tri_num_*sizeof(vec3f)*3);
    for(uint32_t i=0;i<obj_a_tri_num_;i++){
        obj_a_data_[i*3+0]=obj_a_vtxs_[obj_a_tris_[i].id0()];
        obj_a_data_[i*3+1]=obj_a_vtxs_[obj_a_tris_[i].id1()];
        obj_a_data_[i*3+2]=obj_a_vtxs_[obj_a_tris_[i].id2()];
    }
    checkCudaErrors(cudaMalloc((void**)&d_obj_a_data_,obj_a_tri_num_*sizeof(vec3f)*3));
    checkCudaErrors(cudaMemcpy(d_obj_a_data_,obj_a_data_,obj_a_tri_num_*sizeof(vec3f)*3,cudaMemcpyHostToDevice));
    std::cout<<"obj a data generate finish..."<<std::endl;

    if(!is_self_cd){
        obj_b_data_=(vec3f*)malloc(obj_b_tri_num_*sizeof(vec3f)*3);
        for(uint32_t i=0;i<obj_b_tri_num_;i++){
            obj_b_data_[i*3+0]=obj_b_vtxs_[obj_b_tris_[i].id0()];
            obj_b_data_[i*3+1]=obj_b_vtxs_[obj_b_tris_[i].id1()];
            obj_b_data_[i*3+2]=obj_b_vtxs_[obj_b_tris_[i].id2()];
        }
        checkCudaErrors(cudaMalloc((void**)&d_obj_b_data_,obj_b_tri_num_*sizeof(vec3f)*3));
        checkCudaErrors(cudaMemcpy(d_obj_b_data_,obj_b_data_,obj_b_tri_num_*sizeof(vec3f)*3,cudaMemcpyHostToDevice));
        std::cout<<"obj b data generate finish..."<<std::endl;
    }

    checkCudaErrors(cudaMalloc((void**)&d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block));
    h_res=(uint*)malloc(obj_a_tri_num_*sizeof(uint)*threads_per_block);

    std::cout<<"generate tris data finish..."<<std::endl;
}
struct Vec3f{
    double x;
    double y;
    double z;

//    __device__ Vec3f(const double& x,const double &y,const double& z):x(x),y(y),z(z){}
    __device__ Vec3f(double x,double y,double z):x(x),y(y),z(z){}
    __device__ Vec3f():x(0.0),y(0.0),z(0.0){}
    __device__ Vec3f(const Vec3f& v){
        x=v.x;
        y=v.y;
        z=v.z;
    }
    __device__ Vec3f& operator= (const Vec3f& v){
        x=v.x;
        y=v.y;
        z=v.z;
        return *this;
    }
    __device__ Vec3f operator+ (const Vec3f& v) const{
        return Vec3f(x+v.x,y+v.y,z+v.z);
    }

    __device__ Vec3f operator - () const{
        return Vec3f(-x,-y,-z);
    }
    __device__ Vec3f operator- (const Vec3f& v) const{
        return Vec3f(x-v.x,y-v.y,z-v.z);
    }
    __device__ const Vec3f cross(const Vec3f& vec) const {
//        double a=z;
//        double a1=y*vec.z;//double a2=1;double a;a=a1-1;y*vec.z - z*vec.y;
//        double b=z*vec.x - x*vec.z;
//        double c=x*vec.y ;- y*vec.x;
        //return *this;
        return Vec3f(y*vec.z - z*vec.y, z*vec.x - x*vec.z, x*vec.y - y*vec.x);
    }
    __device__ double const dot(const Vec3f& vec) const{
        return x*vec.x+y*vec.y+z*vec.z;
    }
};

__global__ void cuTriContactDetect_v0(tri3f*,tri3f*,
                                      vec3f*,vec3f*,
                                      uint,uint,
                                      uint* res,size_t pitch,double* test);
__global__ void cuTriContactDetect_v1(tri3f* ,tri3f*,
                                      vec3f*,vec3f*,
                                      uint, uint,
                                      uint* res);
__global__ void cuTriContactDetect_v2(vec3f*,vec3f*,uint,uint,uint*);
__global__ void cuTriContactDetect_v3(vec3f*,vec3f*,uint,uint,uint*);
__global__ void cuTriContactDetect_self_v0(tri3f*,vec3f*,uint,uint*);
__global__ void cuTriContactDetect_self_v1(vec3f*,uint,uint*);
__device__ bool cuTriAdjacent(const tri3f& P1,const tri3f& P2);
__device__ bool cuTriAdjacent(vec3f* v1,vec3f* v2);
__device__ bool cuTriContact(const vec3f& P1,const vec3f& P2,const vec3f& P3,const vec3f& Q1,const vec3f& Q2,const vec3f& Q3);
__device__ bool cuSurfaceContact(vec3f ax,
                                 vec3f p1, vec3f p2, vec3f p3);
__device__ bool cuRotationContact(const vec3f& ax,
                                  const vec3f& p1,const vec3f& p2,const vec3f& p3,
                                  const vec3f& q1,const vec3f& q2,const vec3f& q3);
__device__ double cuFmax(double a, double b, double c);
__device__ double cuFmin(double a, double b, double c);


void CollisionDetect::TriContactDetect()
{


////------------------------------------------------------------------------------------
////-----------------Collision Detect between obj a and obj b : method 1----------------
    //一个block负责一个三角形与另一物体所有三角形的判断
    //block数目等于三角形个数较多的数目
//    START_GPU
//        PrepareGPUData();
//        blocks_per_grid=obj_a_tri_num_;
//        std::cout<<blocks_per_grid<<std::endl;
//        cuTriContactDetect_v0<<<blocks_per_grid,threads_per_block>>>
//                (d_obj_a_tris_,d_obj_b_tris_,
//                 d_obj_a_vtxs_,d_obj_b_vtxs_,
//                 obj_a_tri_num_,obj_b_tri_num_,
//                 d_res,pitch,d_test);
//        std::cout<<"finish kernel"<<std::endl;
//        //cudaMemcpy2D(h_res,pitch,d_res,pitch,1024*sizeof(uint),obj_a_tri_num_,cudaMemcpyDeviceToHost);
//        auto err=cudaGetLastError();
//        std::cout<<err<<std::endl;
//        checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));
//        cudaErrorIllegalAddress;
//        //checkCudaErrors(cudaMemcpy(h_test,d_test,sizeof(double)*9,cudaMemcpyDeviceToHost));
//        std::cout<<"finish transfer result"<<std::endl;
//        uint32_t cnt=0;
//        for(int i=0;i<obj_a_tri_num_;i++){
//            uint row=i*threads_per_block;
//            if (h_res[row]>0){
//                //std::cout<<"row is "<<i<<" and num is: "<<h_res[row]<<std::endl;
//                cnt+=h_res[row];
//                for(int j=1;j<=h_res[row];j++){
//                    printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
//                           i,h_res[row+j],
//                           obj_a_tris_[i].id0(),obj_a_tris_[i].id1(),obj_a_tris_[i].id2(),
//                           obj_b_tris_[h_res[row+j]].id0(),obj_b_tris_[h_res[row+j]].id1(),obj_b_tris_[h_res[row+j]].id2());
//                    a_set_.insert(i);
//                    b_set_.insert(h_res[row+j]);
//
//                }
//            }
//        }
//        std::cout<<"gpu cnt is: "<<cnt<<std::endl;
//    END_GPU
////------------------------------------------------------------------------------------
////------------------------------------------------------------------------------------


////------------------------------------------------------------------------------------
////-----------------Collision Detect between obj a and obj b : method 2----------------
//    START_GPU
//    PrepareGPUData();
//    blocks_per_grid=(obj_a_tri_num_+threads_per_block-1)/threads_per_block;
//    cuTriContactDetect_v1<<<blocks_per_grid,threads_per_block>>>
//        (d_obj_a_tris_,d_obj_b_tris_,
//         d_obj_a_vtxs_,d_obj_b_vtxs_,
//         obj_a_tri_num_,obj_b_tri_num_,
//         d_res);
//    std::cout<<"finish kernel"<<std::endl;
//    auto err=cudaGetLastError();
//    std::cout<<err<<std::endl;
//    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));
//    std::cout<<"finish transfer result"<<std::endl;
//    uint32_t cnt=0;
//    uint max_num=0;
//    for(int i=0;i<obj_a_tri_num_;i++){
//        uint row=i*threads_per_block;
//        if (h_res[row]>0){
//            if(h_res[row]>max_num)
//                max_num=h_res[row];
//           // std::cout<<"row is "<<i<<" and num is: "<<h_res[row]<<std::endl;
//            cnt+=h_res[row];
//            for(int j=1;j<=h_res[row];j++){
//                printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
//                           i,h_res[row+j],
//                           obj_a_tris_[i].id0(),obj_a_tris_[i].id1(),obj_a_tris_[i].id2(),
//                           obj_b_tris_[h_res[row+j]].id0(),obj_b_tris_[h_res[row+j]].id1(),obj_b_tris_[h_res[row+j]].id2());
//                a_set_.insert(i);
//                b_set_.insert(h_res[row+j]);
//
//            }
//        }
//    }
//    std::cout<<"gpu cnt is: "<<cnt<<std::endl;
//    std::cout<<"max cd num for one tri is: "<<max_num<<std::endl;
//    cudaErrorIllegalAddress;
//    END_GPU
////------------------------------------------------------------------------------------
////------------------------------------------------------------------------------------


////------------------------------------------------------------------------------------
////-----------------Collision Detect between obj a and obj b : method 3----------------
//    START_GPU
//    GenerateTrisData();
//    blocks_per_grid=obj_a_tri_num_;
//    cuTriContactDetect_v2<<<blocks_per_grid,threads_per_block>>>
//        (d_obj_a_data_,d_obj_b_data_,
//         obj_a_tri_num_,obj_b_tri_num_,
//         d_res);
//    std::cout<<"finish kernel"<<std::endl;
//    auto err=cudaGetLastError();
//    std::cout<<err<<std::endl;
//    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));
//    std::cout<<"finish transfer result"<<std::endl;
//    uint32_t cnt=0;
//    uint max_num=0;
//    for(int i=0;i<obj_a_tri_num_;i++){
//        uint row=i*threads_per_block;
//        if (h_res[row]>0){
//            if(h_res[row]>max_num)
//                max_num=h_res[row];std::cout<<"row is "<<i<<" and num is: "<<h_res[row]<<std::endl;
//            cnt+=h_res[row];
//            for(int j=1;j<=h_res[row];j++){
//                printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
//                           i,h_res[row+j],
//                           obj_a_tris_[i].id0(),obj_a_tris_[i].id1(),obj_a_tris_[i].id2(),
//                           obj_b_tris_[h_res[row+j]].id0(),obj_b_tris_[h_res[row+j]].id1(),obj_b_tris_[h_res[row+j]].id2());
//                a_set_.insert(i);
//                b_set_.insert(h_res[row+j]);
//            }
//        }
//    }
//    std::cout<<"gpu cnt is: "<<cnt<<std::endl;
//    std::cout<<"max cd num for one tri is: "<<max_num<<std::endl;
//    END_GPU
////------------------------------------------------------------------------------------
////------------------------------------------------------------------------------------


////------------------------------------------------------------------------------------
////-----------------Self Collision Detect for single obj a : method 1------------------
    START_GPU
        PrepareGPUData();
        blocks_per_grid=obj_a_tri_num_;
//    std::cout<<"block per grid is: "<<blocks_per_grid<<std::endl;
        cuTriContactDetect_self_v0<<<blocks_per_grid,threads_per_block>>>
                (d_obj_a_tris_,d_obj_a_vtxs_,obj_a_tri_num_,d_res);
        std::cout<<"finish kernel"<<std::endl;
        auto err=cudaGetLastError();
        std::cout<<err<<std::endl;
        checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));
        std::cout<<"finish transfer result"<<std::endl;
        uint32_t cnt=0;
        for(int i=0;i<obj_a_tri_num_;i++){
            uint row=i*threads_per_block;
            if (h_res[row]>0){
                std::cout<<"row is "<<i<<" and num is: "<<h_res[row]<<std::endl;
                cnt+=h_res[row];
                for(int j=1;j<=h_res[row];j++){
                    printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
                           i,h_res[row+j],
                           obj_a_tris_[i].id0(),obj_a_tris_[i].id1(),obj_a_tris_[i].id2(),
                           obj_a_tris_[h_res[row+j]].id0(),obj_a_tris_[h_res[row+j]].id1(),obj_a_tris_[h_res[row+j]].id2());
                    a_set_.insert(i);
                    a_set_.insert(h_res[row+j]);

                }
            }
        }
        std::cout<<"gpu cnt is: "<<cnt<<std::endl;
    END_GPU
////------------------------------------------------------------------------------------
////------------------------------------------------------------------------------------

////------------------------------------------------------------------------------------
////-----------------Self Collision Detect for single obj a : method 2------------------
//    START_GPU
//    GenerateTrisData();
//    blocks_per_grid=obj_a_tri_num_;
////    std::cout<<"block per grid is: "<<blocks_per_grid<<std::endl;
//    cuTriContactDetect_self_v1<<<blocks_per_grid,threads_per_block>>>
//        (d_obj_a_data_,obj_a_tri_num_,d_res);
//    std::cout<<"finish kernel"<<std::endl;
//    auto err=cudaGetLastError();
//    std::cout<<err<<std::endl;
//    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));
//    std::cout<<"finish transfer result"<<std::endl;
//    uint32_t cnt=0;
//    for(int i=0;i<obj_a_tri_num_;i++){
//        uint row=i*threads_per_block;
//        if (h_res[row]>0){
//            std::cout<<"row is "<<i<<" and num is: "<<h_res[row]<<std::endl;
//            cnt+=h_res[row];
//            for(int j=1;j<=h_res[row];j++){
//            printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
//                   i,h_res[row+j],
//                   obj_a_tris_[i].id0(),obj_a_tris_[i].id1(),obj_a_tris_[i].id2(),
//                   obj_a_tris_[h_res[row+j]].id0(),obj_a_tris_[h_res[row+j]].id1(),obj_a_tris_[h_res[row+j]].id2());
//                a_set_.insert(i);
//                a_set_.insert(h_res[row+j]);
//
//            }
//        }
//    }
//    std::cout<<"gpu cnt is: "<<cnt<<std::endl;
//    END_GPU
////------------------------------------------------------------------------------------
////------------------------------------------------------------------------------------
}



__device__ bool test2(double v){
    if (v>0) return true;
    return false;
}
__device__ bool test(Vec3f v){
    double s=v.x+v.y+v.z;
    if (test2(v.x)) return true;
    if (test2(v.y)) return true;
    if (test2(v.z)) return true;
    if (s>0) return true;
    return false;
}
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
__global__ void cuTriContactDetect_v0(tri3f* a_tris,tri3f* b_tris,
                                      vec3f* a_vtxs,vec3f* b_vtxs,
                                      uint a_tris_num,uint b_tris_num,
                                      uint* res,size_t pitch,double* test)
{
    //__shared__ double cache_a_tris[threads_per_block*3];
    //__shared__ double cache_b_tris[threads_per_block*3];
    //__shared__ uint cache_a_vtxs[threads_per_block*3];
    //__shared__ uint cache_b_vtxs[threads_per_block*3];
    __shared__ uint contacted[threads_per_block];
    __shared__ uint cur_contact_num;
    __shared__ Lock lock;

    cur_contact_num=0;

    int gid=threadIdx.x+blockIdx.x*blockDim.x;
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    if (tid==0) lock.init();
    contacted[tid]=0;
    int pass=(b_tris_num+threads_per_block-1)/threads_per_block;
    uint a_tri_x=a_tris[bid].id(0);
    uint a_tri_y=a_tris[bid].id(1);
    uint a_tri_z=a_tris[bid].id(2);

    vec3f a_vtx_x=a_vtxs[a_tri_x];
    vec3f a_vtx_y=a_vtxs[a_tri_y];
    vec3f a_vtx_z=a_vtxs[a_tri_z];
    __syncthreads();
    for(int i=0;i<pass;i++){
        int idx=tid+i*threads_per_block;
        if(idx<b_tris_num){
            uint p_x=b_tris[idx].id(0);
            uint p_y=b_tris[idx].id(1);
            uint p_z=b_tris[idx].id(2);
            vec3f b_vtx_x=b_vtxs[p_x];
            vec3f b_vtx_y=b_vtxs[p_y];
            vec3f b_vtx_z=b_vtxs[p_z];
            if(cuTriContact(a_vtx_x,a_vtx_y,a_vtx_z,b_vtx_x,b_vtx_y,b_vtx_z)){
                lock.lock();
                if (cur_contact_num<threads_per_block){
                    contacted[cur_contact_num]=idx;
                    cur_contact_num++;
                }
                //atomicAdd(&cur_contact_num,1);
                lock.unlock();
            }
        }
        __syncthreads();
    }
    if (tid==0){
        res[bid*threads_per_block]=cur_contact_num;
    }
    if(tid<cur_contact_num && tid<threads_per_block-1)
        res[gid+1]=contacted[tid];

/*        if (bid==0 && tid==0){
            test[0]=a_vtxs[a_tri_x].x;
            test[1]=a_vtxs[a_tri_x].y;
            test[2]=a_vtxs[a_tri_x].z;
            test[3]=a_vtxs[a_tri_y].x;
            test[4]=a_vtxs[a_tri_y].y;
            test[5]=a_vtxs[a_tri_y].z;
            test[6]=a_vtxs[a_tri_z].x;
            test[7]=a_vtxs[a_tri_z].y;
            test[8]=a_vtxs[a_tri_z].z;
        }*/
}
//一个block内一个线程负责一个物体a的三角形
__global__ void cuTriContactDetect_v1(tri3f* a_tris,tri3f* b_tris,
                                      vec3f* a_vtxs,vec3f* b_vtxs,
                                      uint a_tris_num, uint b_tris_num,
                                      uint* res)
{
    //__shared__ Lock lock;
    __shared__ uint contact_num[threads_per_block];
    //first place store contact number
    __shared__ uint contacted[threads_per_block][8];
    // __shared__ double cache_a_tris[threads_per_block][9];
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
    //if (tid==0) lock.init();// one time init for one block
    uint32_t pass=(b_tris_num+threads_per_block-1)/threads_per_block;
    //read a tris from global memory
    tri3f a_tri_id=a_tris[gid];
    vec3f a_vtx_0=a_vtxs[a_tri_id.id0()];
    vec3f a_vtx_1=a_vtxs[a_tri_id.id1()];
    vec3f a_vtx_2=a_vtxs[a_tri_id.id2()];
    //cache_a_tris[tid][0]=a_vtx_0.x;cache_a_tris[tid][1]=a_vtx_0.y;cache_a_tris[tid][2]=a_vtx_0.z;
    //cache_a_tris[tid][3]=a_vtx_1.x;cache_a_tris[tid][4]=a_vtx_1.y;cache_a_tris[tid][5]=a_vtx_1.z;
    //cache_a_tris[tid][6]=a_vtx_2.x;cache_a_tris[tid][7]=a_vtx_2.y;cache_a_tris[tid][8]=a_vtx_2.z;
    contact_num[tid]=0;
    __syncthreads();
    for(uint32_t i=0;i<pass;i++){
        uint pass_t=i*threads_per_block;
        uint idx=tid+pass_t;
        __shared__ double cache_b_tris[threads_per_block][9];
        tri3f b_tri_id=b_tris[idx];
        vec3f b_vtx_0=b_vtxs[b_tri_id.id0()];
        vec3f b_vtx_1=b_vtxs[b_tri_id.id1()];
        vec3f b_vtx_2=b_vtxs[b_tri_id.id2()];
        cache_b_tris[tid][0]=b_vtx_0.x;cache_b_tris[tid][1]=b_vtx_0.y;cache_b_tris[tid][2]=b_vtx_0.z;
        cache_b_tris[tid][3]=b_vtx_1.x;cache_b_tris[tid][4]=b_vtx_1.y;cache_b_tris[tid][5]=b_vtx_1.z;
        cache_b_tris[tid][6]=b_vtx_2.x;cache_b_tris[tid][7]=b_vtx_2.y;cache_b_tris[tid][8]=b_vtx_2.z;
        for(uint32_t j=0;j<threads_per_block;j++){
            uint32_t idx_=(tid+j)%threads_per_block;
            vec3f b_vtx_0_{cache_b_tris[idx_][0],cache_b_tris[idx_][1],cache_b_tris[idx_][2]};
            vec3f b_vtx_1_{cache_b_tris[idx_][3],cache_b_tris[idx_][4],cache_b_tris[idx_][5]};
            vec3f b_vtx_2_{cache_b_tris[idx_][6],cache_b_tris[idx_][7],cache_b_tris[idx_][8]};
            if(cuTriContact(a_vtx_0,a_vtx_1,a_vtx_2,b_vtx_0_,b_vtx_1_,b_vtx_2_)){
                contacted[tid][contact_num[tid]++]=pass_t+idx_;
            }
            //no need sync?
        }
        __syncthreads();
    }
    res[gid*threads_per_block]=contact_num[tid];
    for(uint32_t i=0;i<contact_num[tid];i++){
        res[gid*threads_per_block+i+1]=contacted[tid][i];
    }
}
__global__ void cuTriContactDetect_v2(vec3f* a_data,
                                      vec3f* b_data,uint a_num,uint b_num,uint* res)
{
    __shared__ uint contacted[threads_per_block];
    __shared__ uint contact_num;
    __shared__ Lock lock;
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
    if(tid==0){
        contact_num=0;
        lock.init();
    }

    uint32_t pass=(b_num+threads_per_block-1)/threads_per_block;
    vec3f a_vtx_0=a_data[bid*3+0];
    vec3f a_vtx_1=a_data[bid*3+1];
    vec3f a_vtx_2=a_data[bid*3+2];
    __syncthreads();

    for(uint32_t i=0;i<pass;i++){
        uint32_t idx=tid+i*threads_per_block;
        if(idx<b_num){
            vec3f b_vtx_0=b_data[idx*3+0];
            vec3f b_vtx_1=b_data[idx*3+1];
            vec3f b_vtx_2=b_data[idx*3+2];
            if(cuTriContact(a_vtx_0,a_vtx_1,a_vtx_2,b_vtx_0,b_vtx_1,b_vtx_2)){
                lock.lock();
                if(contact_num<=MAX_CONTACT_NUM){
                    contacted[contact_num]=idx;
//                    res[bid*threads_per_block+1+contact_num]=idx;
                    contact_num++;
                }
                lock.unlock();
            }
        }
        __syncthreads();
    }
    if(tid==0)
        res[bid*threads_per_block]=contact_num;
    if(tid<contact_num && tid<MAX_CONTACT_NUM)
        res[gid+1]=contacted[tid];
}
__global__ void cuTriContactDetect_v3(vec3f* a_data,vec3f* b_data,uint a_num,uint b_num,uint* res)
{

}
__global__ void cuTriContactDetect_self_v0(tri3f* a_tris,vec3f* a_vtxs,uint a_tris_num,uint* res)
{

    __shared__ uint contact_num;
    __shared__ uint contacted[MAX_CONTACT_NUM];
    __shared__ Lock lock;
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
//    printf("kernel\n");
    tri3f a_tri_id=a_tris[bid];
    vec3f a_vtx_0=a_vtxs[a_tri_id.id0()];
    vec3f a_vtx_1=a_vtxs[a_tri_id.id1()];
    vec3f a_vtx_2=a_vtxs[a_tri_id.id2()];
    if(tid==0){
        contact_num=0;
        lock.init();//不能多次初始化
    }
    uint32_t pass=(a_tris_num+threads_per_block-1)/threads_per_block;
    __syncthreads();

    for(uint32_t i=bid/threads_per_block;i<pass;i++){
        uint32_t idx=tid+i*threads_per_block;
        if(idx>bid && idx<a_tris_num){
            tri3f a_tri_id_=a_tris[idx];
            if(!cuTriAdjacent(a_tri_id,a_tri_id_)){
                vec3f a_vtx_0_=a_vtxs[a_tri_id_.id0()];
                vec3f a_vtx_1_=a_vtxs[a_tri_id_.id1()];
                vec3f a_vtx_2_=a_vtxs[a_tri_id_.id2()];
                if(cuTriContact(a_vtx_0,a_vtx_1,a_vtx_2,a_vtx_0_,a_vtx_1_,a_vtx_2_)){
                    lock.lock();
                    if(contact_num<MAX_CONTACT_NUM){
                        contacted[contact_num]=idx;
                        contact_num++;
                    }
                    lock.unlock();
                }
            }
        }
        __syncthreads();
    }
    if(tid==0)
        res[bid*threads_per_block]=contact_num;
    if(tid<contact_num && tid<MAX_CONTACT_NUM)
        res[gid+1]=contacted[tid];
}
__global__ void cuTriContactDetect_self_v1(vec3f* a_data,uint a_num,uint* res)
{
    __shared__ uint contact_num;
    __shared__ uint contacted[MAX_CONTACT_NUM];
    __shared__ Lock lock;
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
    vec3f a_vtxs[3];
    a_vtxs[0]=a_data[bid*3+0];
    a_vtxs[1]=a_data[bid*3+1];
    a_vtxs[2]=a_data[bid*3+2];
    if(tid==0){
        contact_num=0;
        lock.init();//不能多次初始化
    }
    uint32_t pass=(a_num+threads_per_block-1)/threads_per_block;
    __syncthreads();

    for(uint32_t i=bid/threads_per_block;i<pass;i++){
        uint32_t idx=tid+i*threads_per_block;
        if(idx>bid && idx<a_num){
            vec3f a_vtxs_[3];
            a_vtxs_[0]=a_data[idx*3+0];
            a_vtxs_[1]=a_data[idx*3+1];
            a_vtxs_[2]=a_data[idx*3+2];
            if(!cuTriAdjacent(a_vtxs,a_vtxs_)){

                if(cuTriContact(a_vtxs[0],a_vtxs[1],a_vtxs[2],a_vtxs_[0],a_vtxs_[1],a_vtxs_[2])){
                    lock.lock();
                    if(contact_num<MAX_CONTACT_NUM){
                        contacted[contact_num]=idx;
                        contact_num++;
                    }
                    lock.unlock();
                }
            }
        }
        __syncthreads();
    }
    if(tid==0)
        res[bid*threads_per_block]=contact_num;
    if(tid<contact_num && tid<MAX_CONTACT_NUM)
        res[gid+1]=contacted[tid];
}
__device__ bool cuTriAdjacent(const tri3f& P1,const tri3f& P2)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(P1.id(i)==P2.id(j))
                return true;
        }
    }
    return false;
}
__device__ bool cuTriAdjacent(vec3f* v1,vec3f* v2)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(v1[i]==v2[j])
                return true;
        }
    }
    return false;
}
__device__ bool cuTriContact(const vec3f& P1,const vec3f& P2,const vec3f& P3,const vec3f& Q1,const vec3f& Q2,const vec3f& Q3)
{
    vec3f p1;
    vec3f p2=P2-P1;
    vec3f p3=P3-P1;
    vec3f q1=Q1-P1;
    vec3f q2=Q2-P1;
    vec3f q3=Q3-P1;

    vec3f e1=p2-p1;
    vec3f e2=p3-p2;
    vec3f e3=p1-p3;

    vec3f f1=q2-q1;
    vec3f f2=q3-q2;
    vec3f f3=q1-q3;

    vec3f n1=e1.cross(e2);
    vec3f m1=f1.cross(f2);

    vec3f ef11=e1.cross(f1);
    vec3f ef12=e1.cross(f2);
    vec3f ef13=e1.cross(f3);
    vec3f ef21=e2.cross(f1);
    vec3f ef22=e2.cross(f2);
    vec3f ef23=e2.cross(f3);
    vec3f ef31=e3.cross(f1);
    vec3f ef32=e3.cross(f2);
    vec3f ef33=e3.cross(f3);

    if(!cuSurfaceContact(n1,q1,q2,q3)) return false;
    if(!cuSurfaceContact(m1,-q1,p2-q1,p3-q1)) return false;

    if(!cuRotationContact(ef11,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef12,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef13,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef21,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef22,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef23,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef31,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef32,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef33,p1,p2,p3,q1,q2,q3)) return false;

    vec3f g1=e1.cross(n1);
    vec3f g2=e2.cross(n1);
    vec3f g3=e3.cross(n1);

    if(!cuRotationContact(g1,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(g2,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(g3,p1,p2,p3,q1,q2,q3)) return false;

    vec3f h1=f1.cross(m1);
    vec3f h2=f2.cross(m1);
    vec3f h3=f3.cross(m1);
    if(!cuRotationContact(h1,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(h2,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(h3,p1,p2,p3,q1,q2,q3)) return false;

    return true;
}

__device__ bool cuSurfaceContact(vec3f ax,
                                 vec3f p1, vec3f p2, vec3f p3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);

    double mx1 = cuFmax(P1, P2, P3);
    double mn1 = cuFmin(P1, P2, P3);

    if (mn1 > 0.0) return false;
    if (mx1 < 0.0) return false;

    return true;
}

__device__ bool cuRotationContact(const vec3f& ax,
                                  const vec3f& p1, const vec3f& p2, const vec3f& p3,
                                  const vec3f& q1, const vec3f& q2, const vec3f& q3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);
    double Q1 = ax.dot(q1);
    double Q2 = ax.dot(q2);
    double Q3 = ax.dot(q3);

    double mx1 = cuFmax(P1, P2, P3);
    double mn1 = cuFmin(P1, P2, P3);
    double mx2 = cuFmax(Q1, Q2, Q3);
    double mn2 = cuFmin(Q1, Q2, Q3);

    if (mn1 > mx2) return false;
    if (mn2 > mx1) return false;
    return true;
}
__device__ double cuFmax(double a, double b, double c)
{
    double t = a;
    if (b > t) t = b;
    if (c > t) t = c;
    return t;
}
__device__ double cuFmin(double a, double b, double c)
{
    double t = a;
    if (b < t) t = b;
    if (c < t) t = c;
    return t;
}
