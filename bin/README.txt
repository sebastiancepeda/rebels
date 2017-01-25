** Blood vessel segmentation with neural networks

The training and test of a neural network for blood vessel segmentation is made using the DRIVE database of retinal fundus images. 
The neural networks are based on Theano/Lasagne in Python. 

* Dependencies
- pip
- python-dev
- fortran
- BLAS
- Freetype
- OpenCV

sudo apt-get install python-pip python-dev gfortran libatlas-base-dev libfreetype6-dev python-opencv

- The Listed in requirements.txt ("sudo pip install -r requirements.txt"). 

* Database configuration

Is is necessary to download the DRIVE database(http://www.isi.uu.nl/Research/Databases/DRIVE/download.php), and put it in /DRIVE folder.

* Deployment instructions

