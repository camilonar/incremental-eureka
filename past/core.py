import numpy as np

def get_fronteras (sess,x,y,y_,x_input,y_input,delta,nFrontera=None):
    frontera_x = []
    frontera_y = []
    predicciones = []
    input_layer = x_input.shape[1]
    output_layer = y_input.shape[1]
    fronteras = np.empty((0,3))
    for i in range(x_input.shape[0]):
        prueba_pred  = sess.run(y,feed_dict={x: x_input[i].reshape(1,input_layer), y_:y_input[i].reshape(1,output_layer)})
        aux_= np.sort(prueba_pred)
        diferencia = aux_[0][aux_.shape[1]-1]-aux_[0][aux_.shape[1]-2]

        if(diferencia<delta):
            fronteras= np.concatenate((fronteras,[[np.array(x_input[i]),np.array(y_input[i]),diferencia]]),axis=0)
           # frontera_x.append(x_input[i])
           # frontera_y.append(y_input[i])
        predicciones.append([x_input[i],prueba_pred])

    if(nFrontera!=None):
        fronteras = fronteras[fronteras[:, 2].argsort()] #ordenamos las fronteras segun su diferencia
        fronteras = fronteras[:nFrontera]

    fX  = fronteras[:,:1]
    fY   = fronteras[:,1:2]
    frontera_x = np.empty((0,input_layer))
    frontera_y = np.empty((0, output_layer))

    for i in fX:
        frontera_x = np.append(frontera_x, i[0].reshape(1, input_layer), axis=0)

    for i in fY:
        frontera_y = np.append(frontera_y, i[0].reshape(1, output_layer), axis=0)
    return predicciones,frontera_x,frontera_y
	
#TODO: revisar el ordenamiento ascendente o descendente	
def get_centros (sess,x,y,y_,x_input,y_input,delta,nCentro=None):
    centros_x = []
    centros_y = []
    predicciones = []
    input_layer = x_input.shape[1]
    output_layer = y_input.shape[1]
    centros = np.empty((0,3))
    for i in range(x_input.shape[0]):
        prueba_pred  = sess.run(y,feed_dict={x: x_input[i].reshape(1,input_layer), y_:y_input[i].reshape(1,output_layer)})
        aux_= np.sort(prueba_pred)
        diferencia = aux_[0][aux_.shape[1]-1]-aux_[0][aux_.shape[1]-2]
        if(diferencia > delta):
            centros = np.concatenate((centros,[[np.array(x_input[i]),np.array(y_input[i]),diferencia]]),axis=0)
        predicciones.append([x_input[i],prueba_pred])

    if(nCentro!=None):
        centros = centros[centros[:, 2].argsort()[::-1]] #ordenamos los centros segun su diferencia
        centros = centros[:nCentro]

    cX = centros[:,:1]
    cY = centros[:,1:2]
    centros_x = np.empty((0,input_layer))
    centros_y = np.empty((0, output_layer))

    for i in cX:
        centros_x = np.append(centros_x, i[0].reshape(1, input_layer), axis=0)

    for i in cY:
        centros_y = np.append(centros_y, i[0].reshape(1, output_layer), axis=0)
    return predicciones,centros_x,centros_y