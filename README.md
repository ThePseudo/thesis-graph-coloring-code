# Graph Coloring

## Build

You should build the project to compile with the right flags for your CUDA-enabled device.
In the following script, replace **\<XX\>** with the Compute Capability of your device.
The default option in `GC_GENCODE_SM70`. You can disable it with ```-DGC_GENCODE_SM70=OFF```.
If you are unsure, enable the option `CUDA_AUTODETECT_GENCODE`

```console
mkdir build
cd build
cmake -DGC_GENCODE_SM<XX>=ON ..
make
```

### Timings
Streams are expressed in milliseconds, no-streams in seconds.
