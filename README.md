# SliceNet MEC Application (GPU Version)

GPU version the MEC application for the SliceNet project

## Getting Started

Before starting, you'll need to edit the facial_recognition.py. Here you will fill out a patient record for each person who will use the application. You will also need to provide
a portrait photograph for each user (see obama.jpg for example)

### Prerequisites

Will need to be installed in this order:

* [Python 2.7](https://www.python.org/download/releases/2.7/) - Python x64 interpreter
* [Boost](https://www.boost.org/) - Boost C++ Libraries*
* [Dlib](https://github.com/davisking/dlib) - Machine Vision / AI Library*
* [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - Library for Support Vector Machines*

Follow instructions from each third party to install the prerequisites.

* Compile the boost libraries with Python bindings.
* Compile Dlib with CUDA (A Nvidia GPU is required for compiling CUDA with Dlib, non-GPU version is also available)
* Python MAT expects the LIBSVM python files to be located at "..\libsvm\python" unless changed by the user.

### Running the program

Once all the above prerequisites are installed, navigate to the root folder of the application, then start the server.

```
python server.py
```

You are than ready to launch the Gateway application !