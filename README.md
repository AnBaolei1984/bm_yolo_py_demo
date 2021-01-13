# bm_yolo_py_demo 使用说明

使用到的API主要可以参考文档 sdk路径/documents/Sophon_Inference_zh.pdf

python版本支持 3.5 - 3.8

1. 安装Sophon Inference的python包

   a) 在sdk目录find -name *\sophon*\.whl
      
     选择对应的whl，se5、sm5选择./lib/sail/python3/arm_pcie里的轮子，sc5板卡选择./lib/sail/python3/pcie目录里的轮子。根据自己的python版本选择对应的轮子。
     
    比如在sc5、python3.5就选择./lib/sail/python3/pcie/py35/sophon-2.2.0-py3-none-any.whl

   b）安装 
     
     先卸载之前可能安装的包

     pip3 uninstall sophon

     安装

     pip3 install sophon sophon-2.2.0-py3-none-any.whl

2. 编译c++ so

    cd yolo_postprocess_lib/

    1）se5、sm5编译

      make -f Makefile.arm

    2）sc5 编译

      make -f Makefile.pcie

    编译成功后将编译出来的libYoloPostProcess.so放到和det_yolo_bmcv.py的同级目录

3. 执行demo

    python3 det_yolo_bmcv.py --bmodel yolov4_coco_416_int8.bmodel --input test2.mp4

    执行后会在生成的result_imgs目录里看到结果图片
      
