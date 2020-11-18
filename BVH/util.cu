//
// Created by lidan on 13/11/2020.
//

#include "util.cuh"
#include <vector>





BVH::BVH(BVHInfo *bvhInfos, int size) {
    int allnodes = 0;
    root = SplitBuild(bvhInfos,0,size,&allnodes) ;
    treearraysize = getDeep(root,0);
    int allsize = getsize(root) ;
    std::cout<<" all deep of this tree is:"<<treearraysize<<std::endl ;
    std::cout<<"all size of this tree is:"<<allsize<<std::endl ;
    result = new TreeBoundBox[pow(2,treearraysize)] ;
    getArrayBox(root,0) ;
}

void BVH::getArrayBox(BVHNode* root,int index)
    {
        result[index].box = root->node_box ;
        result[index].id = root->nid ;
        if(root->children[0]!= nullptr) getArrayBox(root->children[0],index*2+1) ;
        if(root->children[1]!= nullptr) getArrayBox(root->children[1],index*2+2) ;
//    if(root->children[0]== nullptr&&root->children[1]== nullptr)
//    {
//        result[index].id = root->nid  ;
//        std::cout<<result[index].id<<std::endl;
//    }
}

int BVH::getDeep(BVHNode* root,int deep)
{
    if(root== nullptr) return deep ;
    int dep1 = getDeep(root->children[0],deep+1) ;
    int dep2 = getDeep(root->children[1],deep+1) ;
    return dep1>dep2?dep1:dep2 ;
}

int BVH::getsize(BVHNode* root)
{
    if(root==nullptr) return 0 ;
    int size1 = getsize(root->children[0]) ;
    int size2 = getsize(root->children[1]) ;

    if(size1==0 && size2==0)
    {
        if(root->nid<0) std::cout<<"shit happened"<<std::endl ;
//        std::cout<<"this is "<<root->nid<<std::endl ;
        return 1;
    }
    return size1+size2 ;
}

int di = 0 ;
bool compare(const BVHInfo* &left,BVHInfo* &right){
    return left->centroid[di]>right->centroid[di] ;
}


BVHNode* BVH::SplitBuild(BVHInfo* infos, int begin, int end, int *allnodes) {

    if(end-begin==0){ return nullptr ;}
//    static int size  = 0 ;
    BVHNode* node = new BVHNode ;
    (*allnodes)++ ;
    BoundBox box;

    for (int i = begin; i < end; ++i)
        box.merge(infos[i].box) ;
    int number = end - begin;

    if(number == 1)
    {
        node->leafNode(infos[begin].id_, number, box);
        maps[begin] = node ;
        if(begin<50)
        std::cout<<"begin is "<<begin<<" box :"<<node->node_box.pmin[0]<<" "
                                                <<node->node_box.pmin[1]<<" "
                                                <<node->node_box.pmin[2]<<" "<<std::endl ;
//        std::cout<<"begin is "<<begin<<" end is "<<end<<std::endl ;
        return node ;
    }else
    {
        BoundBox centerBox ;
        for(int i = begin ; i<end ; i++ )
        {
            centerBox.merge(infos[i].centroid) ;
        }
        int dimension =centerBox.MaximumExtent() ;
        int mid =(end - begin)/2 + begin ;


        std::sort(&infos[begin],&infos[end-1]+1,[dimension](const BVHInfo &left,BVHInfo &right)->bool
        {
            return left.centroid[dimension]>right.centroid[dimension] ;
        }) ;

//        double pmid ;
//        //if all the vertexs are coincided
//        if(abs(centerBox.pmin[dimension]-centerBox.pmax[dimension])<1e-10)
//        {
////            node->leafNode(begin,number,box) ;
////            return node ;
//            node->InNode(dimension,SplitBuild(infos,begin,mid,allnodes),SplitBuild(infos,mid,end,allnodes)) ;
//        }// leaf node's number are great than zero
//        else{
//            pmid = centerBox.getmidP()[dimension];
//            BVHInfo* midptr = std::partition(&infos[begin],&infos[end-1]+1,[dimension,pmid](const BVHInfo& pi){
//                return pi.centroid[dimension] < pmid ;
//            }) ;
//            mid = midptr - &infos[0];
////
////            di = dimension ;
////            std::sort(&infos[begin],&infos[end-1]+1,compare) ;
//        }
////        std::cout<<"begin is "<<begin<<" end is "<<end<<std::endl ;
        node->InNode(dimension,SplitBuild(infos,begin,mid,allnodes),SplitBuild(infos,mid,end,allnodes)) ;

    }

    return node ;
}

