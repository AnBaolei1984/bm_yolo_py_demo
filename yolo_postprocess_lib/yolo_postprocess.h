/* Copyright 2019-2024 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#ifndef _YOLO_POSTPROCESS_HPP_
#define _YOLO_POSTPROCESS_HPP_

#include <vector>
#include <string>

struct ObjRect {
  unsigned int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
};

struct yolov3_box {
  float x;
  float y;
  float w;
  float h;
};

struct yolov3_DetectRect {
  int left;
  int right;
  int top;
  int bot;
  float score;
  int category;
};

struct detection {
  yolov3_box bbox;
  int classes;
  float *prob;
  float *mask;
  float objectness;
  int sort_class;
};

struct layer {
  int batch;
  int total;
  int n, c, h, w;
  int out_n, out_c, out_h, out_w;
  int classes;
  int inputs, outputs;
  int *mask;
  float *biases;
  float *output;
  float *output_gpu;
};

typedef struct __tag_st_process_info{
  float* biases_;
  int* masks_;
  int biases_num_;
  int classes_num_;
  int anchor_num_;
  float threshold_nms_;
  float threshold_prob_;
  int fm_size_[6];
  int net_w_ ;
  int net_h_;
}st_process_info;

extern "C" {
  void process(
         long inputs[],
         char* net_shape,
         char* input_shape,
         int input_tensor_num,
         float thresh_hold,
         int class_num,
         char* anchor_biases,
         char* masks,
         int anchor_biases_num,
         int img_width,
         int img_height,
         int top_k,
         char* results,
         char* obj_num);
}

#endif /* _YOLO_POSTPROCESS_HPP_ */
