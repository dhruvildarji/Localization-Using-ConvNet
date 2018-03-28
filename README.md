# FindingPhone

Author_DHRUVIL_A_DARJI_
Project Title: Phone Detecton 
Used Method: Convolution Neural Network
Used Library: Scikit NeuralNetworks  http://scikit-neuralnetwork.readthedocs.io

Step 1:
Run train_phone_finder.py
	python train_phone_finder.py -i PATH OF FOLDER

It will generate pickle file in the same folder of the file train_phone_finder. I will use this file to test the images.
I didn't have access to GPU. 
But I have included Theano module, which will execute the script on your GPU.
If you have GPU available, then it should be detected and run the code in your GPU if you have nvcc installed properly.
It took me probably half an hour to train this network.
I have included pickle file named "ready_pickle.pkl" in the folder.
If you don't want to waste your time to train the model, then you can put "ready_pickle.pkl" file in find_phone.py directly. 
In order to do that, you will need to change name of the pickel file in Line 18 in find_phone.py 

Step 2:
Run find_phone.py
	python find_phone.py -i PATH OF A TEST IMAGE
	
It will return coordinates of the phone.

Approach:

==> Classification approach has been used to find the phone.
==> Cut the image in several pieces. By considering the Threshold value (0.05), the size of the small pieces (36,48) of the image has been determine.
==> Find Center , Left Top and Right Bottom Coordinates of Each pieces of the image.
==> Label the Images containing the phone as "1", the Images without phone as "0".
==> To balance the number of images with 0s and number of images with 1s, duplicate the images of label "1" in training dataset.
==> Train the network with 4 Convolution layers and softmax output layer.
==> Learning rate is set to 0.002
==> Number of Epoches are 30
==> Batch size is equal to 25
==> Normalize the network with respect to Batches.
==> Validation set is 20 % of total images

Limitation:

==> Did not access for the GPU for fast computation
==> Tensorflow doesn't support python 2.7 with windows. I am very flexible with Linux and Macintosh. But I don't own it.
==> Small Dataset
==> Learning rate was uncontrollable with this Function. I had to make it fix for entire training.


Possible Solutions:

==> Alternatively, this problem can be formulated as an object localization problem, and solved using a CNN with regression header, where the input is a image that containing one instance of object of interest, the output should be the location of that object.
==> Then we can train the CNN model with the given images and ground-truth locations (center x and y values). The current data set with 100+ images is probably not enough to train an adequate model. With enough training data set, this approach could achieve smaller error of x and y values.
