1. How to build the TensorFlow C++ API from sources

    1.1 Install appropriate version of protobuf
    
    1.2 Get sources
    
        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v2.x (replace with the appropriate version)
        
    1.3 Build the sources
    
        bazel build tensorflow:tensorflow_cc
        
    1.4 Generate header
    
        bazel build tensorflow:install_headers
        
    1.5 Create symlinks to the shared libraries
    
        libtensorflow_framework.so -> libtensorflow_framework.so.2.13.0
        libtensorflow_framework.so.2 -> libtensorflow_framework.so.2.13.0
        libtensorflow_cc.so -> libtensorflow_cc.so.2.13.0
        libtensorflow_cc.so.2 -> libtensorflow_cc.so.2.13.0

    File locations:
    
        Headers located in the <repo_root>/bazel-bin/tensorflow/include
        libtensorflow_cc.so and libtensorflow_framework.so located in the <repo_root>/bazel-bin/tensorflow

