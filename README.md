# Hand-Written-Digit-Recognition
Pattern recognition using Feedforward Fully Connected method
<br /><br />

# About This Project
In this project we used MNIST dataset (http://yann.lecun.com/exdb/mnist/) which includes 28*28 array of 0 to 255 as the luminance.<br />
We created 2 hidden layers each has 16 neurons.<br /><br />
![This is an image](https://s4.uupload.ir/files/screenshot_from_2021-12-17_12-27-51_qwdg.png)

### step 1 (read.py):<br />
Extracting MNIST dataset which includes 60000 data in Train Set ( 'train-images-idx3-ubyte' , 'train-labels-idx1-ubyte' ) and 10000 data in Test Set ( 't10k-images-idx3-ubyte' , 't10k-labels-idx1-ubyte' ) and we plotted some of them to ensure it works properly.<br /><br />

### step 2 (feedForward.py):<br />
In each layer calculating next layer neuron with this formula : __a(L+1) = sigmoid (W(L+1)*a(L) + b(L+1))__ <br />
First randomly choose 100 images of Train Set and initialize Weight matrix with random numbers and Bias matrix as Zero Matrix, then calculate output for these images and accuracy of model<br /><br />

### step 3 (backPropagation.py):<br />
Using Gradient Descent to minimize Cost function with batch_size=10, alpha=1, epoch_num=20 then plotting average of Costs for different epochs<br /><br />

# Built With
- [python](https://www.python.org/) <br /><br />

# Getting Started
### Prerequisites
- put '_train-images-idx3-ubyte_, _train-labels-idx1-ubyte_, _t10k-images-idx3-ubyte_, _t10k-labels-idx1-ubyte_' in your project path
- numpy <br />
    `pip install numpy`
- matplotlib <br />
    `pip install pip install matplotlib`
- math <br />
    `pip install python-math`
- random <br />
    `pip install random`
    
<br /><br />
# License
Distributed under the MIT License. See `LICENSE.txt` for more information
<br /><br />

# Contact
rezaie.somayeh79@gmail.com
