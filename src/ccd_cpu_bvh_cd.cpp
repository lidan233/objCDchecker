//
// Created by lidan on 17/11/2020.
//

#include "ccd_cpu_bvh_cd.h"



inline static std::vector<int> isInteract(TreeBoundBox* bvh,BoundBox* box)
{
    std::vector<int> stack ;
    std::vector<int> result ;
    stack.push_back(0) ;

    while(stack.size()!=0)
    {
        int t = stack.at(stack.size()-1) ;
        stack.pop_back() ;
        if(bvh[t].box.interacted(*box))
        {

            if(bvh[t].id>=0){ result.push_back(bvh[t].id); continue; }
            if (bvh[2*t+1].id!=-2)   stack.push_back(2*t+1) ;
            if (bvh[2*t+2].id!=-2)   stack.push_back(2*t+2) ;

        }
    }
    return result ;
}


inline static std::vector<int> isInteract_tree(BVHNode* bvh,BoundBox* box)
{
    std::vector<BVHNode*> stack ;
    std::vector<int> result ;
    stack.push_back(bvh) ;

    while(stack.size()!=0)
    {
        BVHNode* t = stack.at(stack.size()-1) ;
        stack.pop_back() ;
        if(t->node_box.interacted(*box))
        {

            if(t->nid>=0){ result.push_back(t->nid); continue; }
            if(t->children[0]!= nullptr) stack.push_back(t->children[0]) ;
            if(t->children[1]!= nullptr) stack.push_back(t->children[1]) ;

        }
    }
    return result ;
}


ccd_cpu_bvh_cd::ccd_cpu_bvh_cd()
{
}

void ccd_cpu_bvh_cd::init()
{
    collusion.clear() ;
}

std::set<int>* ccd_cpu_bvh_cd::checkCDByBVH(gpu_mesh* mesh1,gpu_mesh* mesh2)
{
    return nullptr ;
}


set<int>* ccd_cpu_bvh_cd::checkSelfCDByBVH(gpu_mesh* mesh1)
{

    TreeBoundBox* bvhs = mesh1->get_cpu_bvh();
    BVHNode* root = mesh1->get_cpu_bvh_tree() ;
    vec3fcu* tris_data = mesh1->cpu_get_alldata() ;
    vec3icu* tris_id = mesh1->cpu_get_faceid();
    map<int,BVHNode*> maps = mesh1->get_cpu_bvh_map() ;


    int count = 0 ;

#pragma omp parallel for num_threads(24)
    for(int i = 0 ; i< mesh1->get_face_size()*3;i+=3)
    {
        vec3fcu point1 = tris_data[i] ;
        vec3fcu point2 = tris_data[i+1] ;
        vec3fcu point3 = tris_data[i+2] ;
        vec3icu id1 = tris_id[i/3]  ;

        BoundBox box = BoundBox(point1,point2,point3) ;
//        std::cout<<box.interacted(box)<<std::endl ;
//        std::cout<<box.interacted(maps[i/3]->node_box)<<std::endl ;
        std::vector<int> t_face = isInteract(bvhs,&box) ;
//        std::vector<int> t_face = isInteract_tree(root,&box) ;
//        std::cout<<"there is "<<t_face.size()<<" collusion"<<" for face "<<i/3<<std::endl ;
        for(auto j = t_face.begin() ; j!=t_face.end();j++)
        {
            if(*j<(i/3)) continue;
            vec3fcu point4 = tris_data[(*j)*3] ;
            vec3fcu point5 = tris_data[(*j)*3+1] ;
            vec3fcu point6 = tris_data[(*j)*3+2] ;
            vec3icu id2 = tris_id[(*j)] ;

            bool cons = true ;
            for(int g = 0 ; g< 3;g++)
            {
                for(int h = 0 ; h <3 ;h++)
                {
                    if(id1[g]==id2[h])
                    {
                        cons = false ;
                        break ;
                    }
                }
            }

            if(cons && cutri_contact(point1,point2,point3,point4,point5,point6) == true)
            {
                collusion.insert(i/3) ;
                collusion.insert(*j) ;

                std::cout<<"bvh cpu check there is collusion between ("<<i/3<<" ,"<<*j<<")"<<std::endl ;
#pragma omp atomic
                count += 1 ;
            }
        }
    }
    std::cout<<" bvh cpu check there is collusion happened "<<count<<" collusion."<<std::endl ;
    std::cout<<" bvh cpu check there is collusion face "<<collusion.size()<<" "<<std::endl;

    return &collusion ;

}