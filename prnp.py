import numpy as np

frontera = np.empty((0,3))



print(frontera.shape)

frontera = np.concatenate((frontera ,np.array([[np.array([1,2]),2,3]])),axis=0)
frontera = np.concatenate((frontera ,np.array([[np.array([1,2]),4,5]])),axis=0)
frontera = np.concatenate((frontera ,np.array([[np.array([0,1]),1,0]])),axis=0)
frontera = np.concatenate((frontera ,np.array([[np.array([54,1]),4,5]])),axis=0)


frontera = frontera[frontera[:,1].argsort()]
print(frontera,"\n")

fronteraX   = frontera[:,:1]

print(fronteraX[1:2][0])

arrayfinal = np.empty((0,2))

for i in fronteraX:
    arrayfinal  = np.append(arrayfinal,i[0].reshape(1,2),axis=0)

