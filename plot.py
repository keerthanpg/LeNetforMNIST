import matplotlib.pyplot as plt
import numpy
import pickle
import testLeNet
import cnn_lenet

layer2_30 = pickle.load( open( "2ndlayerafter10000.pickle", "rb" ) )
print(layer2_30)

layers=testLeNet.get_lenet()
print(layers[3])

output=cnn_lenet.pooling_layer_forward(layer2_30, layers[3])
layer=output['data'][:,1].reshape((12,12,20))


f,axarr = plt.subplots(4,5)
for i in range(4):
	for j in range(5):
		axarr[i][j].imshow(layer[:,:,i*4+j], cmap='gray')

plt.show()


'''
layer=layer2_30['data'][:,1].reshape((12,12,20))
print(layer.shape)
print(numpy.amax(layer2_30['data']), numpy.amin(layer2_30['data']))

#plt.imshow(layer, cmap='gray')
#plt.show()


f, axarr = plt.subplots(4,5)
for i in range(4):
	for j in range(5):
		axarr[i][j].imshow(layer[:,:,i*4+j], cmap='gray')


plt.show()'''