** Blood vessel segmentation with neural networks

The training and test of a neural network for blood vessel segmentation is made using the DRIVE database of retinal fundus images. 
The neural networks are based on Theano/Lasagne in Python. 

* Dependencies
- pip (sudo apt-get install python-pip)
- python-dev (sudo apt-get install python-dev)
- fortran (sudo apt-get install gfortran)
- BLAS (sudo apt-get install libatlas-base-dev)
- Freetype (sudo apt-get install libfreetype6-dev)
- OpenCV (sudo apt-get install python-opencv)
- Lasagne (sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt)
- The Listed in requirements.txt ("sudo pip install -r requirements.txt"). 

* Database configuration

Is is necessary to download the DRIVE database(http://www.isi.uu.nl/Research/Databases/DRIVE/download.php), and put it in /DRIVE folder.

* Deployment instructions

