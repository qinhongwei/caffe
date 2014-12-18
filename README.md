This is a version for my won work on deepfish.
There are also `winmex` branch and `windows` branch by Zhiheng Niu.
The `master` version is developed by BVLC.
# Windows Installation
This is not the original [Caffe Readme](https://github.com/BVLC/caffe/blob/master/README.md) but an installation guide for windows version.

#### Want to run first before build by yourself?
You can download the windows x64 [standalone package](https://dl.dropboxusercontent.com/u/3466743/caffe-vs2012/standalone.7z) and run directly on MNIST dataset.

#### Prerequisites
You may need the followings to build the code:
- Windows 64-bit
- MS Visual Studio 2012
- CUDA toolkit 6.0
- Other dependencies which you can directly download from [here](http://dl.dropboxusercontent.com/u/3466743/caffe-vs2012/dependency-20140804.7z).

#### Build Steps
Currently it can be built by VS2012 for x64 flatform only. This is because the dependencies mentioned above is cross-compiled to support x64 only. If you want to build on 32bit windows, you need to rebuild your own 3rd-party libraries.
- Check out the code and switch to *windows* branch
- Download the dependency file and extract the folders inside to project root directory.
- Include any .cpp you want to build in the `./tools` directory to MainCaller.cpp.
- Open the solution file in `./build/MSVC`
- Switch build target to x64 platform (Both debug and release are OK).
- Build the code and you may find the `./bin/MainCaller.exe`

#### Train MNIST dataset
- Suppose you choose to build train_net.cpp which is the default one in MainCaller.cpp
- If you do not have GPU, please change it to CPU in `lenet_solver.prototxt`
- Goto directory `./examples/mnist`
- Double click `get_mnist_leveldb.bat` to download the dataset in leveldb format.
- Double click `train_lenet.bat` to see the training progress .

#### Known Issues
- It takes obvious longer time when you compile for the first time. Therefore please refrain from using `clean & rebuild`.
- I have trained on ImageNet with this windows porting as well. The speed is much slower than the one built on Ubuntu. 20 iterations take 79s on Windows, whereas same number of iterations take about 30s on Ubuntu (on GTX Titan).

#### Bug Report
- Please create new issues in [github](https://github.com/niuzhiheng/caffe/issues) if you find any bug.
- If you have new pull requests, they are very welcome.
- Before you do that, you may wish to read this [wiki page](https://github.com/niuzhiheng/caffe/wiki) for more information.
