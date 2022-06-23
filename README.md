# Lightweight Multidimensional Adaptive Sampling for GPU Ray Tracing
Sources codes of the <a href="https://jcgt.org/published/0011/02/05/">Lightweight Multidimensional Adaptive Sampling for GPU Ray Tracing</a> project. 
We extended Optix samples to support the proposed parallel multidimensional sampling and reconstruction. In particular, we added five new samples: optixMotionBlur, optixDepthOfField, optixAmbientOcclusion, optixPathTracer, and optixDirectLighting.

## Compilation
We compiled the project with Visual Studio 2019 (x64), but it should work also with other compilers using CMake.

## Usage
There are three sample scenes in SDK/data: pool, cornell-box, and chess. We use env file format for the configuration. Besides scene configuration, we can also configure sampling:
```
Sampler {
    mdas true # use mdas or qmc
    samples 8 # number of saples
}

Mdas { # mdas parameters (see aper for details
    scaleFactor 1 
    alpha 0.25
    bitsPerDim 1
    extraImgBits 8 
}
```

We simply use the env file as argument to run the sample:
```
./optixMotionBlur.exe ../../../data/pool/pool.env
./optixDepthOfField.exe ../../../data/chess/chess.env
./optixPathTracer.exe ../../../data/cornell-box/cornell-box.env
```

There test scripts in SDK/data/Scripts that we used to generate the paper results.

## License
The additional code is released into the public domain. 

## Citation
If you use this code, please cite <a href="https://jcgt.org/published/0011/02/05/">the paper</a>:
```
@Article{Meister2022,
  author = {Daniel Meister and Toshiya Hachisuka},
  title = {{Lightweight Multidimensional Adaptive Sampling for GPU Ray Tracing}},
  journal = {Journal of Computer Graphics Techniques (JCGT)},
  volume = {11},
  number = {2},
  pages = {91--112},
  year = {2022},
}
```
