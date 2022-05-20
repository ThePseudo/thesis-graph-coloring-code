# Graph Coloring

## Build

You should build the project to compile with the right flags for your CUDA-enabled device.
In the following script, replace **\<XX\>** with the Compute Capability of your device.
If you are unsure, enable the option `CUDA_AUTODETECT_GENCODE`

```console
mkdir build
cd build
cmake -DGC_GENCODE_SM<XX> ..
make
```