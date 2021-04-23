# import torch
# import numpy as np
# import torch.nn as nn
# import random
# from sklearn import preprocessing
# k = 3
# mu = 3
# sigma = 0.5
# utility = np.random.normal( mu , sigma , k*k*k*k ).reshape((k,k,k,k))

# def nesh_step(Acc,normal_index,reduce_index):
#     # normals =nn.Parameter(torch.Tensor([[0.5,0.4,0.1],[0.6,0.2,0.2],[0.3,0.5,0.2],[0.1,0.9,0.0]]))
#     # reduces =nn.Parameter(torch.Tensor([[0.5,0.4,0.1],[0.6,0.2,0.2],[0.3,0.5,0.2],[0.1,0.9,0.0]]))
#     #random.seed(1)
#     #p = np.random.rand(4,k)#Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
#     for i in range(len(normal_index)):
#         utility[normal_index[i][0],normal_index[i][1],normal_index[i][2],normal_index[i][3]] = Acc[i]
#     for i in range(len(reduce_index)):
#         utility[reduce_index[i][0],reduce_index[i][1],reduce_index[i][2],reduce_index[i][3]] = Acc[i]
#     epoch = 0
#     iteration = 0
#     min_u_min = [100, epoch]
#     # normal_alpha = []
#     # reduce_alpha = []
#     alpha = []
#     for iteration in range(2):
#         p_min = []
#         epoch=0
#         while epoch < 20000:
#             random.seed(epoch)
#             p = np.random.rand(4,k)
#             p =[ p[i]/p.sum(1)[i] for i in range(4)]
#             #p = [preprocessing.normalize(p[i]) for i in range(4)]
#             #p_ = [np.outer(p[i],p[j]) for i in range(4) for j in range(4) if i < j]#6
#             p__ = [np.outer(np.outer(p[0],p[1]),p[2]),np.outer(np.outer(p[0],p[1]),p[3]),np.outer(np.outer(p[0],p[2]),p[3]),np.outer(np.outer(p[1],p[2]),p[3])]#4
#             u1 = [np.concatenate((utility[i]),axis=0) for i in range(k)]
#             u1_sum = [np.tensordot(u1[i],p__[0]) for i in range(k)]
#             u1_min = min(u1_sum)
#             u2 = [np.concatenate((utility[:,i]),axis=0) for i in range(k)]
#             u2_sum = [np.tensordot(u2[i],p__[1]) for i in range(k)]
#             u2_min = min(u2_sum)
#             u3 = [np.reshape([np.append([],[utility[i,j][0]]) for i in range(k) for j in range(k)],(k*k,k)),
#                 np.reshape([np.append([],[utility[i,j][1]]) for i in range(k) for j in range(k)],(k*k,k)),
#                 np.reshape([np.append([],[utility[i,j][2]]) for i in range(k) for j in range(k)],(k*k,k))]
#             u3_sum = [np.tensordot(u3[i],p__[2]) for i in range(k)]
#             u3_min = min(u3_sum)
#             u4 = [np.reshape([np.append([],[utility[i,j][:,0]]) for i in range(k) for j in range(k)],(k*k,k)),
#                 np.reshape([np.append([],[utility[i,j][:,1]]) for i in range(k) for j in range(k)],(k*k,k)),
#                 np.reshape([np.append([],[utility[i,j][:,2]]) for i in range(k) for j in range(k)],(k*k,k))]
#             u4_sum = [np.tensordot(u3[i],p__[3]) for i in range(k)]
#             u4_min = min(u4_sum)
#             u_min = sum([u1_min,u2_min,u3_min,u4_min])
#             if u_min < min_u_min[0] :
#                 p_min =  p
#                 min_u_min = [u_min, epoch]
#             epoch += 1
#             # print(p_min)
#             # print(min_u_min)
#         alpha.append(p_min)
#     print(alpha)
#     return nn.Parameter(torch.Tensor(alpha[0])),nn.Parameter(torch.Tensor(alpha[1]))



import torch
import numpy as np
import torch.nn as nn
import random
from sklearn import preprocessing

def nesh_step(Acc,normal_index,reduce_index):
    k = 3
    mu = 3
    sigma = 0.5
    utility = np.random.normal( mu , sigma , k*k*k*k ).reshape((k,k,k,k))

    # normals =nn.Parameter(torch.Tensor([[0.5,0.4,0.1],[0.6,0.2,0.2],[0.3,0.5,0.2],[0.1,0.9,0.0]]))
    # reduces =nn.Parameter(torch.Tensor([[0.5,0.4,0.1],[0.6,0.2,0.2],[0.3,0.5,0.2],[0.1,0.9,0.0]]))
    #random.seed(1)
    #p = np.random.rand(4,k)#Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
    
    epoch = 0
    iteration = 0
    min_u_min = [100, epoch]
    # normal_alpha = []
    # reduce_alpha = []
    alpha = []
    while iteration <2:
        index = [normal_index,reduce_index][iteration]
        for i in range(len(index)):
            utility[index[i][0],index[i][1],index[i][2],index[i][3]] = Acc[i]
        p_min = []
        epoch = 0
        print("=====",epoch)
        while epoch < 20000:
            random.seed(epoch)
            p = np.random.rand(4,k)
            p =[ p[i]/p.sum(1)[i] for i in range(4)]
            #p = [preprocessing.normalize(p[i]) for i in range(4)]
            #p_ = [np.outer(p[i],p[j]) for i in range(4) for j in range(4) if i < j]#6
            p__ = [np.outer(np.outer(p[0],p[1]),p[2]),np.outer(np.outer(p[0],p[1]),p[3]),np.outer(np.outer(p[0],p[2]),p[3]),np.outer(np.outer(p[1],p[2]),p[3])]#4
            u1 = [np.concatenate((utility[i]),axis=0) for i in range(k)]
            u1_sum = [np.tensordot(u1[i],p__[0]) for i in range(k)]
            u1_min = min(u1_sum)
            u2 = [np.concatenate((utility[:,i]),axis=0) for i in range(k)]
            u2_sum = [np.tensordot(u2[i],p__[1]) for i in range(k)]
            u2_min = min(u2_sum)
            u3 = [np.reshape([np.append([],[utility[i,j][0]]) for i in range(k) for j in range(k)],(k*k,k)),
                np.reshape([np.append([],[utility[i,j][1]]) for i in range(k) for j in range(k)],(k*k,k)),
                np.reshape([np.append([],[utility[i,j][2]]) for i in range(k) for j in range(k)],(k*k,k))]
            u3_sum = [np.tensordot(u3[i],p__[2]) for i in range(k)]
            u3_min = min(u3_sum)
            u4 = [np.reshape([np.append([],[utility[i,j][:,0]]) for i in range(k) for j in range(k)],(k*k,k)),
                np.reshape([np.append([],[utility[i,j][:,1]]) for i in range(k) for j in range(k)],(k*k,k)),
                np.reshape([np.append([],[utility[i,j][:,2]]) for i in range(k) for j in range(k)],(k*k,k))]
            u4_sum = [np.tensordot(u3[i],p__[3]) for i in range(k)]
            u4_min = min(u4_sum)
            u_min = sum([u1_min,u2_min,u3_min,u4_min])
            if u_min < min_u_min[0] :
                p_min = p
                min_u_min = [u_min, epoch]
            epoch += 1
            # print(p_min)
            # print(min_u_min)
        alpha.append(p_min)
        print(alpha,epoch)
        iteration += 1
        print(iteration)
    return nn.Parameter(torch.Tensor(alpha[0])),nn.Parameter(torch.Tensor(alpha[1]))
