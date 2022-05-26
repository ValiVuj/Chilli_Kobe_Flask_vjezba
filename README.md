# Chilli_Kobe_Flask_vjezba

This small project includes a binary classification(classifying photos of my dog and my cat) and labeling them-Kobe(dog) vs Chilli(cat). I've developed an interest in computer vision while participating on a 6 months course of programming AI. I wanted to make a small application for my project so I've actually been learning Flask, Jinja and Werkzeug throughout this project. Firstly, I've made a small neural network myself but since I didn't get good results, I've used a pretrained network.	
	Here, I've worked with: Tensorflow, Matplotlib, Numpy, Open CV.
	I've tried pretrained networks such as: ResNet50, Inceptionv3, VGG16, but I've had best results with VGG19 with weights from ImageNet.
	Optimizers I've tried: Adam, SGD with Momentum, RMSProp, but I've chosen Adadelta due to the best results after training.
	Loss function is Categorical CrossEntropy.
	Test accuracy: 78.18182110786438
	This project would've worked better if I had had more photos of my pets, therefore there is a place for improvement since my cat is not recognized as cat all the time.
