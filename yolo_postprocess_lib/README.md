#opencv f32/int8 demo for nntoolchain
#note!!! in docker test
#before test, read NeuralNetwork/YOLOv3_object/model/README.md

##compile
```
    make -f Makefile
```

##test f32
```
    make -f Makefile f32 loops=1
```
**you can find out-01_vehicle_1.jpg

##test int8
```
    make -f Makefile int8 loops=1
```
**you can find out-00_vehicle_1.jpg
