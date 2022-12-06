import cv2
import numpy as np
from sklearn.neighbors import KernelDensity
import itertools
import sys
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path,edmonds_karp,preflow_push,boykov_kolmogorov
import time

BG=1
FG=2
# global fg_blue
# global bg_red

def computeBeta(image_copy,num_neighbors):
    num_edges=0
    beta=0
    for y in range(image_copy.shape[0]):
        for x in range(image_copy.shape[1]):
            neighbors = getNNeighbors(image_copy.shape[0:2],num_neighbors,(y,x),None)
            num_edges+=len(neighbors)
            for dest in neighbors:
                diff= image_copy[y,x]-image_copy[dest[0],dest[1]]
                beta+=np.linalg.norm(diff)
    beta = 1/(2*beta/num_edges)
    return beta

def getMasks(bg_pix,fg_pix,image_copy):
    temp= cv2.copyTo(image_copy,None,None)
    mask =mask_denoise = np.zeros(image_copy.shape[0:2])
    if(len(fg_pix)!=0):
        fg_ind=tuple(zip(*list(fg_pix)))
        mask[fg_ind]=1
        temp[fg_ind]=fg_blue
    temp[np.where(mask==0)]=bg_red
    # for y in range(image_copy.shape[0]):
    #     for x in range(image_copy.shape[1]):
    #         if((y,x) in fg_pix):
    #             temp[y,x]=fg_blue
    #             mask[y,x]=1
    #         elif((y,x) in bg_pix or seed[y,x]==BG):
    #             temp[y,x]=bg_red

    if not HSV:
        image_copy_denoise = cv2.fastNlMeansDenoisingColored(temp,None,31,31,7,21)
    else:
        image_copy_denoise = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(temp, cv2.COLOR_HSV2BGR),None,31,31,7,21)
    
    hsv=cv2.cvtColor(image_copy_denoise, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60,0,0])
    upper_blue = np.array([170,255,255])
    mask_denoise=cv2.inRange(hsv,lower_blue,upper_blue)
    mask_denoise[np.where(mask_denoise>0)]=1

    return mask,mask_denoise,temp,image_copy_denoise

def str_to_coords(str):
    coords= str[1:len(str)-1].split(',')
    return (int(coords[0]),int(coords[1]))

def partition_to_coords(partition):
    res=set()
    for str in partition:
        res.add(str_to_coords(str))
    return res

def getNNeighbors(image_shape,num_neighbors,coords,r):
    changes=[i for i in range(-num_neighbors,num_neighbors+1)]
    deltas = list(itertools.product(changes,changes))
    neighbors =[]
    for delta in deltas:
        dest=(coords[0]+delta[0], coords[1]+delta[1])
        if dest != coords:
            neighbors.append(dest)
    def filter_neighbor(dest):
        if r!=None:
            return (dest[0]>r[1] and dest[0]<r[3] and dest[1]>r[0] and dest[1]<r[2])\
            and (not (dest[0]<0 or dest[0]>=image_shape[0] or dest[1]<0 or dest[1]>=image_shape[1]))
                 
        else:
            return not (dest[0]<0 or dest[0]>=image_shape[0] or dest[1]<0 or dest[1]>=image_shape[1])
        
    return list(filter(filter_neighbor,neighbors))

def getNNeighborsDict(image_shape,num_neighbors,r):
    neighbors={}
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            neighbors[(y,x)]= getNNeighbors(image_shape,num_neighbors,(y,x),r)
    return neighbors




class Node:
    def __init__(self,coords) -> None:
        self.edges =[]
        self.coords=coords
        self.inOverflowQ=False

class Edge:
    def __init__(self,start,end,capacity,inGraph=False) -> None:
        self.capacity=capacity
        self.flow =0
        self.source=start
        self.destination=end
        self.revEdge=None
        self.inGraph=inGraph
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and self.source==__o.source and self.destination == __o.destination
    def __hash__(self) -> int:
        return hash((self.source,self.destination))

class FlowNetwork:
    def __init__(self)-> None:
        self.source = Node((-1,-1))
        self.sink = Node((-2,-2))
        self.nodes = {(-1,-1):self.source, (-2,-2):self.sink} # pixel coord to Node map
    def addEdge(self, start, end,capacity):
        if(capacity==0):
            #print("Did not add edge with capacity 0.")
            return True
        if(capacity<0):
            #print(f"Did not add edge {start},{end} with capacity {capacity}!")
            return False
        self.addNode(start)
        self.addNode(end)
        edge= Edge(start,end,capacity,True)
        rev_edge = Edge(end,start,0)
        edge.revEdge = rev_edge
        rev_edge.revEdge=edge
        self.nodes[start].edges.append(edge)
        self.nodes[end].edges.append(rev_edge)
        return True

    def addNode(self, coords):
        if(coords not in self.nodes):
            self.nodes[coords] = Node(coords)

    def augmentingPathDFS(self):
        visited =set()
        stack=[(self.source.coords,[])]
        path=[]
        while(len(stack)!=0):
            coords, path = stack.pop()
            n = self.nodes[coords]
            if(coords not in visited):
                visited.add(coords)
                if(n.coords == self.sink.coords):
                    print('path found')
                    return path
                for edge in n.edges:
                    residual_capacity = edge.capacity -edge.flow
                    if(edge.destination not in visited and residual_capacity>0):
                        stack.append((edge.destination,path+[(edge,residual_capacity)]))
        return None
    
    def augmentingPathBFS(self):
        visited =set()
        path={}
        min_res={self.source.coords:sys.maxsize}
        q=[self.source.coords]
        visited.add(self.source.coords)
        while(len(q)!=0):
            coords = q.pop(0)
            n = self.nodes[coords]
            for edge in n.edges:
                residual_capacity = edge.capacity - edge.flow
                if(edge.destination not in visited and residual_capacity>0):
                    path[edge.destination]=(n.coords,edge)
                    visited.add(edge.destination)
                    min_res[edge.destination]= min(min_res[n.coords],residual_capacity)
                    if(edge.destination == self.sink.coords):
                        print('path found')
                        return path, min_res[edge.destination]
                    else:
                        q.append(edge.destination)
        return None,None

    def FordFulkerson(self):
        path,min_cap_on_path =self.augmentingPathBFS()
        max_flow=0
        while(path != None):
            max_flow+=min_cap_on_path
            v=self.sink.coords
            while(v!=self.source.coords):
                parent=path[v]
                parent_coords,edge=parent
                edge.flow+=min_cap_on_path
                edge.revEdge.flow-=min_cap_on_path
                v=parent_coords
            path,min_cap_on_path=self.augmentingPathBFS()
        return max_flow
    
    def pushRelabel(self):

        #initialize pre-flow
        height={key: 0 for key in self.nodes.keys()}
        #pushable={key:set() for key in self.nodes.keys()}
        height[self.source.coords]=len(self.nodes)
        excess_flow ={key: 0 for key in self.nodes.keys()}
        inQ = {key:False for key in self.nodes.keys()}
        q=[]
        #vertices = [k for k in self.nodes.keys() if k!=self.source.coords and k!=self.sink.coords]
        for edge in self.source.edges:
            edge.flow=edge.capacity
            edge.revEdge.flow = -edge.capacity
            excess_flow[edge.destination]= edge.capacity
            excess_flow[self.source.coords] -= edge.capacity
            if(edge.destination!=self.sink.coords):
                q.append(edge.destination)
                inQ[edge.destination]=True
        def getOverFlowingNode():
            for v in self.nodes.values():
                if(excess_flow[v.coords]>0 and v.coords!=self.sink.coords and v.coords!=self.source.coords):
                    return v.coords
            return None
        def processNode(node_coords):
            if excess_flow[node_coords]==0:
                return
            node =self.nodes[node_coords]
            for edge in node.edges:
                res_cap=edge.capacity-edge.flow
                if excess_flow[node_coords]>0 :
                    if res_cap>0 and height[node_coords]==height[edge.destination]+1:
                        push(edge)
                else:
                    break
            relabel(node)
            
        def push(edge):
            res_cap=edge.capacity-edge.flow
            min_flow= min(res_cap, excess_flow[edge.source])
            # if(i%100==0):
            #     print('pushing', min_flow)
            if(edge.inGraph):
                edge.flow+=min_flow
            else:
                edge.revEdge.flow-=min_flow
            excess_flow[edge.destination]+=min_flow
            excess_flow[edge.source]-=min_flow
            
            if(not inQ[edge.destination] and edge.destination!=self.source.coords and edge.destination!=self.sink.coords):
                q.append(edge.destination)
                inQ[edge.destination]=True

        def relabel(node):
            heights=[height[edge.destination] for edge in node.edges if edge.capacity-edge.flow>0]
            # if(i%100==0):
            #     print([(edge.destination,edge.capacity,edge.flow,height[edge.destination]) for edge in node.edges])
            if excess_flow[node.coords]>0 and len(heights)>0:
                # if (i%100==0):
                #   print('relabel')
                height[node.coords] = 1+ min(heights)
                q.append(node_coords)
                inQ[node_coords]=True

        #v= getOverFlowingNode()
        i=0
        while(len(q)>0):
            # if(i%100==0):
            #     print("i",i,'len q',len(q))
            #node_coords=vertices[i]
            node_coords =q.pop(0)
            inQ[node_coords]=False
            #print("excess flow before",node_coords,excess_flow[node_coords],'height',prev_height)
            processNode(node_coords)
            # if prev_height != height[node_coords]:
            #     i=0
            #     vertices.insert(0,vertices.pop(i))
            # else:
            #     i+=1
            #print("excess flow after",node_coords,excess_flow[node_coords],'height',height[node_coords])
            #v=getOverFlowingNode()
            i+=1
        return sum([edge.flow for edge in self.source.edges]), excess_flow

    def MinCut(self):
        visited =set()
        min_cut=0
        BFSq=[self.source.coords]
        visited.add(self.source.coords)
        while (len(BFSq) > 0):
            node_coords = BFSq.pop(0)
            node= self.nodes[node_coords]
            for edge in node.edges:
                if edge.capacity-edge.flow > 0 and edge.destination not in visited:
                    visited.add(edge.destination)
                    BFSq.append(edge.destination)

        for node in self.nodes.values():
            for edge in node.edges:
                res_cap=edge.capacity -edge.flow
                if edge.source in visited and edge.destination  not in visited and  res_cap==0:
                    min_cut+=edge.flow
        return min_cut, visited, set(self.nodes.keys()).difference(visited) # min_cut cost, source pixels, sink pixels

# Graph Cut
# Make cdfs from background and foreground intensity seeds
def getCDFs(seed,mask,image_copy):
    if mask is not None:
        fg =image_copy[np.where(np.logical_or(seed==FG, np.logical_and(seed==0, mask==1)))]
        bg= image_copy[np.where(np.logical_or(seed==BG, np.logical_and(seed==0, mask==0)))]
    else:
        fg=image_copy[np.where(seed==FG)]
        bg= image_copy[np.where(seed == BG)]
        
    # for y in range(seed.shape[0]):
    #     for x in range(image_copy.shape[1]):
    #         if(seed[y,x] == FG):
    #             fg.append(image_copy[y,x])
    #         elif(mask is not None):
    #             if(mask[y,x]==1):
    #                 fg.append(image_copy[y,x])
    #             else:
    #                 bg.append(image_copy[y,x])
    #         elif(seed[y,x]==BG):
    #             bg.append(image_copy[y,x])
    #print(fg)
    fg_cdf= KernelDensity(kernel='gaussian').fit(fg.reshape(-1,3)).score_samples
    bg_cdf=KernelDensity(kernel='gaussian').fit(bg.reshape(-1,3)).score_samples
    return fg_cdf,bg_cdf

def nLinkWeight(I_p,I_q,mask,p_coords,q_coords,beta,gamma):
    if(mask is not None and mask[p_coords]==mask[q_coords]):
        return 0
    # if sigma != None:
    #     return gamma * np.round(np.exp(-1*(np.linalg.norm(I_p-I_q))**2/(2*sigma^2)) * 1/np.sqrt((p_coords[0]-q_coords[0])**2 + (p_coords[1]-q_coords[1])**2))
    diff=(I_p-I_q)
    return gamma * np.exp(-beta*(np.linalg.norm(diff)**2)) #* 1/np.sqrt((p_coords[0]-q_coords[0])**2 + (p_coords[1]-q_coords[1])**2)

def tLinkWeights(I_p,cdf_fg,cdf_bg, seed,weights):
    if seed==FG:
        return sys.maxsize,0
    elif seed==BG:
        return 0, sys.maxsize
    return -weights[0]*np.round(cdf_bg([I_p])[0]), -weights[1]*np.round(cdf_fg([I_p])[0])


def createGraph(seed,_mask,image_copy,num_neighbors,sigma,beta,gamma,r,weights=(0.95,1.2),neighborsDict=None):
    # weights for source link is 0.95 and sink link is 1.2 for corpus callosum. sigma (doesn't really matter)
    # num_neighbors in multiples of 8
    start=time.time()
    FN=FlowNetwork()
    cdf_fg, cdf_bg = getCDFs(seed,_mask,image_copy)
    # add nodes
    for y in range(image_copy.shape[0]):
        for x in range(image_copy.shape[1]):
            if r!=None and (not (y>r[1] and y<r[3] and x>r[0] and x<r[2])):
                continue
            source_weight, sink_weight = tLinkWeights(image_copy[y,x],cdf_fg,cdf_bg,seed[y,x],weights)
            FN.addEdge((-1,-1),(y,x),source_weight)
            FN.addEdge((y,x),(-2,-2),sink_weight)
            neighbors = getNNeighbors(image_copy.shape[0:2],num_neighbors,(y,x),r) if neighborsDict ==None else neighborsDict[(y,x)]
            for dest in neighbors:
                FN.addEdge((y,x),dest, nLinkWeight(image_copy[y,x],image_copy[dest[0],dest[1]],_mask,(y,x),dest,sigma,beta,gamma))
    end=time.time()   
    print(f'{end-start} seconds to create graph')    
    return FN 

    
def createNXGraph(seed,_mask,image_copy,num_neighbors,beta,gamma,r,weights=(0.95,1.2),neighborsDict=None):
    # weights for source link is 0.95 and sink link is 1.2 for corpus callosum. sigma (doesn't really matter)
    # num_neighbors is number of square matrix dimensions
    start=time.time()
    FN=nx.DiGraph()
    cdf_fg, cdf_bg = getCDFs(seed,_mask,image_copy)
    # add nodes
    for y in range(image_copy.shape[0]):
        for x in range(image_copy.shape[1]):
            if r!=None and (not (y>r[1] and y<r[3] and x>r[0] and x<r[2])):
                continue
            source_weight, sink_weight = tLinkWeights(image_copy[y,x],cdf_fg,cdf_bg,seed[y,x],weights)
            if source_weight!=0:
                FN.add_edge(str((-1,-1)),str((y,x)),capacity=source_weight)
            if sink_weight!=0:
                FN.add_edge(str((y,x)),str((-2,-2)),capacity=sink_weight)
            neighbors = getNNeighbors(image_copy.shape[0:2],num_neighbors,(y,x),r) if neighborsDict ==None else neighborsDict[(y,x)]
            for dest in neighbors:
                cap=nLinkWeight(image_copy[y,x],image_copy[dest[0],dest[1]],_mask,(y,x),dest,beta,gamma)
                if cap!=0:
                    FN.add_edge(str((y,x)),str(dest), capacity=cap)
    end=time.time()
    print(f'{end-start} seconds to create graph')    
    return FN

def iterativeGraphcut(num_iter,tol,seed,image_copy,_num_neighbors,gammas,r,_weights=[(0.95,1.2)],NX=False,denoise=False,hsv=False):
    global fg_blue
    global bg_red
    global HSV
    if not hsv:
        fg_blue=[225,0,0]
        bg_red=[0,0,255]
    else:
        fg_blue=[120, 255, 225]
        bg_red=[0,255,255]
    HSV=hsv
    start=time.time()
    mask=None
    mask_denoise= None
    beta = computeBeta(image_copy,sum(_num_neighbors)//len(_num_neighbors))
    print('beta=',beta)
    i=0
    prevFlow= 0
    flows=[]
    masks=[]
    mask_denoises=[]
    FNs=[]
    while(i<num_iter):
        if(len(gammas)!=0):
            gamma = gammas.pop(0)
        if(len(_weights)!=0):
            weights=_weights.pop(0)
        if(len(_num_neighbors)!=0):
            num_neighbors = _num_neighbors.pop(0)
        neighbors_dict = getNNeighborsDict(image_copy.shape[0:2],num_neighbors,r)
        flow, mask,mask_denoise,fg_pix,bg_pix,FN= graphCutV1(seed,mask if not denoise else mask_denoise
                                                ,image_copy,num_neighbors,beta,gamma,r,weights,NX=NX,neighborsDict=neighbors_dict)
        flows.append(flow)
        masks.append(mask)
        mask_denoises.append(mask_denoise)
        FNs.append(FN)
        print(f'Flow delta: {abs(prevFlow-flow)}, Flow {flow}')
        if prevFlow!=0 and abs(prevFlow-flow) <tol:
            break
        prevFlow=flow
        i+=1
    end=time.time()
    print(f'{end-start} seconds to run iterative graphCut')
    return flows,masks,mask_denoises,FNs

def graphCutV1(seed,mask_,image_copy,num_neighbors,beta,gamma,r,weights=(0.95,1.2),NX=False,neighborsDict=None):
    if not NX:
        FN= createGraph(seed,mask_,image_copy,num_neighbors,beta,gamma,r,weights)
        start=time.time()
        FN.pushRelabel()
        min_cut,fg_pix,bg_pix=FN.MinCut()
        mask,mask_denoise,_,_=getMasks(bg_pix,fg_pix,image_copy)
        end=time.time()
        print(f'{end-start} seconds to run graphCutV1')
        return min_cut,mask,mask_denoise,fg_pix,bg_pix

    FN= createNXGraph(seed,mask_,image_copy,num_neighbors,beta,gamma,r,weights,neighborsDict)
    start=time.time()
    min_cut,partition=nx.minimum_cut(FN,str((-1,-1)),str((-2,-2)),flow_func=preflow_push)
    fg_pix,bg_pix=partition
    fg_pix=partition_to_coords(fg_pix)
    bg_pix=partition_to_coords(bg_pix)
    mask,mask_denoise,_,_=getMasks(bg_pix,fg_pix,image_copy)
    end=time.time()
    print(f'# of fg and bg pix {len(fg_pix)},{len(bg_pix)}')
    print(f'{end-start} seconds to run graphCutV1')
    return min_cut,mask,mask_denoise,fg_pix,bg_pix,FN
        