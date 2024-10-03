import numpy as np

DTYPE = np.floatc

cdef struct array_id:
    int [:] tab

cdef struct array :
    float [:] tab



cdef array_id Neighbors(float[:] x, float [:,:] x_train, int n_neighbours):
    df=np.zeros((len(x_train),1), dtype=DTYPE)
    cdef int[:] df_view = df
    
    for i in range(x_train.shape[0]):
        df_view[i] = np.sqrt((x[0]-x_train[i,0])**2+(x[1]-x_train[i,1])**2)
    
    return np.argsort(df_view)[:n_neighbours]


cdef array KNN_classification(train,x_test,n_neighbours):
    DF=np.zeros((x_test.shape[0],3))
    DF[:,:2]=x_test
    class_train = train[:,0]
    class_train=class_train.astype(int)
    x_train = train[:,1:]
    
    for i in range(x_test.shape[0]):
        #On calcule les distances entre chaque ligne de x_test avec les données dans x_train
        Neighbors_id = Neighbors(x_test[i,:],x_train,n_neighbours)
        value_counts = np.bincount(class_train[Neighbors_id])
        DF[i,2]=np.argmax(value_counts)
    return DF