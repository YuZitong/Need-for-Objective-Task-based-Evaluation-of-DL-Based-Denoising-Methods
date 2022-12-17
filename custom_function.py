import numpy as np

def MultiLDpooled(IS,IN,U):
    # test statistics
    tS = np.zeros(len(IS))
    tN = np.zeros(len(IN))
    # input images
    IS = np.transpose(np.asmatrix(IS), [1,0]) # (262144,N)
    IN = np.transpose(np.asmatrix(IN), [1,0])  # (262144,N)

    for i in np.arange(IS.shape[-1]):
        IS_training = np.delete(IS,i,axis=1)
        # channalize input images
        vS = np.matmul(U,IS_training) # (5,N-1)
        vN = np.matmul(U,IN) # (5,N)
        delta_g_bar = np.matmul(U,np.mean(IS_training,axis=1)-np.mean(IN,axis=1)) # (5,262144)(262144,1) -> (5,1)
        vData = np.concatenate((vS,vN),axis=1) # (5,2N-1)
        K = np.cov(vData) # (5,5) covariance matrix
        eta = 0.5*(np.matmul(np.matmul(np.transpose(np.mean(vN,axis=1),[1,0]), np.linalg.inv(K)),np.mean(vN,axis=1))\
            - np.matmul(np.matmul(np.transpose(np.mean(vS,axis=1),[1,0]), np.linalg.inv(K)),np.mean(vS,axis=1)))
        vS_t = np.matmul(U,IS[:,i])
        tS[i] = np.matmul(np.matmul(np.transpose(delta_g_bar,[1,0]), np.linalg.inv(K)),vS_t)+eta

    for i in np.arange(IN.shape[-1]):
        IN_training = np.delete(IN,i,axis=1)
        vS = np.matmul(U,IS) # (5,N)
        vN = np.matmul(U,IN_training) # (5,N-1)
        delta_g_bar = np.matmul(U,np.mean(IS,axis=1)-np.mean(IN_training,axis=1)) # (5,262144)(262144,1) -> (5,1)
        vData = np.concatenate((vS,vN),axis=1) # (5,2N-1)
        K = np.cov(vData) # (5,5)
        eta = 0.5*(np.matmul(np.matmul(np.transpose(np.mean(vN,axis=1),[1,0]), np.linalg.inv(K)),np.mean(vN,axis=1))\
            - np.matmul(np.matmul(np.transpose(np.mean(vS,axis=1),[1,0]), np.linalg.inv(K)),np.mean(vS,axis=1)))
        vN_t = np.matmul(U,IN[:,i])
        tN[i] = np.matmul(np.matmul(np.transpose(delta_g_bar,[1,0]), np.linalg.inv(K)),vN_t)+eta

    return tS,tN
