/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

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
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include "yolo_postprocess.h"
using namespace std;

static int nms_comparator(const void* pa, const void* pb) {
  detection a = *reinterpret_cast<const detection*>(pa);
  detection b = *reinterpret_cast<const detection*>(pb);
  float diff = 0;
  if (b.sort_class >= 0) {
    diff = a.prob[b.sort_class] - b.prob[b.sort_class];
  } else {
    diff = a.objectness - b.objectness;
  }
  return diff < 0 ? 1 : -1;
}

static float box_iou(yolov3_box a, yolov3_box b) {
  float area1 = a.w * a.h;
  float area2 = b.w * b.h;
  float wi = std::min((a.x + a.w / 2), (b.x + b.w / 2))
             - std::max((a.x - a.w / 2), (b.x - b.w / 2));
  float hi = std::min((a.y + a.h / 2), (b.y + b.h / 2))
             - std::max((a.y - a.h / 2), (b.y - b.h / 2));
  float area_i = std::max(wi, 0.0f) * std::max(hi, 0.0f);
  return area_i / (area1 + area2 - area_i);
}

static void do_nms_sort(
    detection* dets,
    int        total,
    int        classes,
    float      thresh) {
  int i, j, k;
  k = total - 1;
  for (i = 0; i <= k; ++i) {
    if (dets[i].objectness == 0) {
      detection swap = dets[i];
      dets[i] = dets[k];
      dets[k] = swap;
      --k;
      --i;
    }
  }
  total = k + 1;
  for (k = 0; k < classes; ++k) {
    for (i = 0; i < total; ++i) {
      dets[i].sort_class = k;
    }
    qsort(dets, total, sizeof(detection), nms_comparator);
    for (i = 0; i < total; ++i) {
      if (dets[i].prob[k] == 0) continue;
      yolov3_box a = dets[i].bbox;
      for (j = i + 1; j < total; ++j) {
        yolov3_box b = dets[j].bbox;
        if (box_iou(a, b) > thresh) {
          dets[j].prob[k] = 0;
        }
      }
    }
  }
}

static layer make_yolo_layer(
    const st_process_info& proc_info,
    int    mask_index,
    int    batch,
    int    w,
    int    h,
    int    n,
    int    total,
    int    classes) {
  layer l = {0};
  l.n = n;
  l.total = total;
  l.batch = batch;
  l.h = h;
  l.w = w;
  l.c = n * (classes + 4 + 1);
  l.out_w = l.w;
  l.out_h = l.h;
  l.out_c = l.c;
  l.classes = classes;
  l.inputs = l.w * l.h * l.c;
  l.biases = reinterpret_cast<float*>(calloc(total * 2, sizeof(float)));
  for (int i = 0; i < total * 2; ++i) {
    l.biases[i] = proc_info.biases_[i];
  }

  l.mask = reinterpret_cast<int*>(calloc(n, sizeof(int)));
  for (int i = 0; i < l.n; ++i) {
      l.mask[i] = proc_info.masks_[mask_index];
      mask_index += 1;
  }
  l.outputs = l.inputs;
  l.output = reinterpret_cast<float*>(calloc(batch* l.outputs, sizeof(float)));
  return l;
}

static void free_yolo_layer(layer l) {
  if (NULL != l.biases) {
    free(l.biases);
    l.biases = NULL;
  }
  if (NULL != l.mask) {
    free(l.mask);
    l.mask = NULL;
  }
  if (NULL != l.output) {
    free(l.output);
    l.output = NULL;
  }
}

static int entry_index(
    layer    l,
    int      batch,
    int      location,
    int      entry) {
  int n = location / (l.w * l.h);
  int loc = location % (l.w * l.h);
  return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1)
         + entry * l.w * l.h + loc;
}

static void forward_yolo_layer(
    const float*    input,
    layer           l) {
  memcpy(l.output, reinterpret_cast<const float *>(input),
         l.outputs * l.batch * sizeof(float));
}

static int yolo_num_detections(
    layer     l,
    float     thresh) {
  int i, n;
  int count = 0;
  for (i = 0; i < l.w * l.h; ++i) {
    for (n = 0; n < l.n; ++n) {
      int obj_index = entry_index(l, 0, n * l.w * l.h + i, 4);
      if (l.output[obj_index] > thresh) {
        ++count;
      }
    }
  }
  return count;
}

static int num_detections(
    std::vector<layer> layers_params,
    float              thresh) {
  unsigned int i;
  int s = 0;
  for (i = 0; i < layers_params.size(); ++i) {
    layer l = layers_params[i];
    s += yolo_num_detections(l, thresh);
  }
  return s;
}

static detection* make_network_boxes(
    std::vector<layer>    layers_params,
    float                 thresh,
    int*                  num) {
  layer l = layers_params[0];
  int nboxes = num_detections(layers_params, thresh);
  if (num) {
    *num = nboxes;
  }
  detection* dets =
      reinterpret_cast<detection*>(calloc(nboxes, sizeof(detection)));
  for (int i = 0; i < nboxes; ++i) {
    dets[i].prob = reinterpret_cast<float*>(calloc(l.classes, sizeof(float)));
  }
  return dets;
}

static void correct_yolo_boxes(
    detection*    dets,
    int           n,
    int           w,
    int           h,
    int           netw,
    int           neth,
    int           relative) {
  int new_w = 0;
  int new_h = 0;
  if (((float)netw / w) < ((float)neth / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  new_w = netw;
  new_h = neth;
  for (int i = 0; i < n; ++i) {
    yolov3_box b = dets[i].bbox;
    b.x = (b.x - (netw - new_w) / 2. / netw)
        / ((float)new_w / netw);
    b.y = (b.y - (neth - new_h) / 2. / neth)
        / ((float)new_h / neth);
    b.w *= (float)netw / new_w;
    b.h *= (float)neth / new_h;
    if (!relative) {
      b.x *= w;
      b.w *= w;
      b.y *= h;
      b.h *= h;
    }
    dets[i].bbox = b;
  }
}

static yolov3_box get_yolo_box(
     float*    x,
     float*    biases,
     int       n,
     int       index,
     int       i,
     int       j,
     int       lw,
     int       lh,
     int       w,
     int       h,
     int       stride) {
  yolov3_box b;
  if (x[index + 1 * stride] > 1 || x[index + 1 * stride] < 0) {
    std::cout << x[index + 1 * stride] << std::endl;
  }
  b.x = (i + x[index + 0 * stride]) / lw;
  b.y = (j + x[index + 1 * stride]) / lh;
  b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
  b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
  return b;
}

static int get_yolo_detections(
    layer         l,
    int           w,
    int           h,
    int           netw,
    int           neth,
    float         thresh,
    int*          map,
    int           relative,
    detection*    dets) {
  float* predictions = l.output;
  int count = 0;
  for (int i = 0; i < l.w * l.h; ++i) {
    int row = i / l.w;
    int col = i % l.w;
    for (int n = 0; n < l.n; ++n) {
      int obj_index = entry_index(l, 0, n * l.w * l.h + i, 4);
      float objectness = predictions[obj_index];
      if (objectness <= thresh) {
        continue;
      }
      int box_index = entry_index(l, 0, n * l.w * l.h + i, 0);
      dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n],
                                      box_index, col, row, l.w, l.h,
                                      netw, neth, l.w * l.h);
      dets[count].objectness = objectness;
      dets[count].classes = l.classes;
      for (int j = 0; j < l.classes; ++j) {
        int class_index = entry_index(l, 0, n * l.w * l.h + i, 4 + 1 + j);
        float prob = objectness * predictions[class_index];
        dets[count].prob[j] = (prob > thresh) ? prob : 0;
      }
      ++count;
    }
  }
  correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
  return count;
}

static void fill_network_boxes(
    const st_process_info& proc_info,
    std::vector<layer>    layers_params,
    int                   w,
    int                   h,
    float                 thresh,
    float                 hier,
    int*                  map,
    int                   relative,
    detection*            dets) {
  for (size_t j = 0; j < layers_params.size(); ++j) {
    layer l = layers_params[j];
    int count = get_yolo_detections(l, w, h, proc_info.net_w_, proc_info.net_h_,
                                    thresh, map, relative, dets);
    dets += count;
  }
}

static detection* get_network_boxes(
    const st_process_info& proc_info,
    std::vector<layer>    layers_params,
    int                   img_w,
    int                   img_h,
    float                 thresh,
    float                 hier,
    int*                  map,
    int                   relative,
    int*                  num) {
  detection* dets = make_network_boxes(layers_params, thresh, num);
  fill_network_boxes(proc_info, layers_params, img_w, img_h,
                     thresh, hier, map, relative, dets);
  return dets;
}

static detection* get_detections(
    std::vector<float*>    blobs,
    const st_process_info& proc_info,
    int                    img_w,
    int                    img_h,
    int*                   nboxes) {
  int classes = proc_info.classes_num_;
  float thresh = proc_info.threshold_prob_;
  float hier_thresh = proc_info.threshold_prob_;
  float nms = proc_info.threshold_nms_;

  std::vector<layer> layers_params;
  layers_params.clear();
  for (unsigned int i = 0; i < blobs.size(); ++i) {
    layer l_params = make_yolo_layer(proc_info, proc_info.anchor_num_ * i, 1, proc_info.fm_size_[2 * i],
        proc_info.fm_size_[2 * i + 1], proc_info.anchor_num_, proc_info.biases_num_ / 2, classes);
    layers_params.push_back(l_params);
    forward_yolo_layer(blobs[i], l_params);  /* blobs[i] host_mem data */
  }
  detection *dets = get_network_boxes(proc_info, layers_params, img_w, img_h,
                                      thresh, hier_thresh, 0, 1, nboxes);
  /* release layer memory */
  for (unsigned int index = 0; index < layers_params.size(); ++index) {
    free_yolo_layer(layers_params[index]);
  }
  if (nms) {
    do_nms_sort(dets, (*nboxes), classes, nms);
  }
  return dets;
}

static void free_detections(detection *dets, int nboxes) {
  for (int i = 0; i < nboxes; ++i) {
    free(dets[i].prob);
  }
  free(dets);
}

static int set_index(
    int    w,
    int    h,
    int    num,
    int    classes,
    int    batch,
    int    location,
    int    entry) {
  int n = location / (w * h);
  int loc = location % (w * h);
  int c = num * (classes + 4 + 1);
  int output = w * h * c;
  return batch * output + n * w * h * (4 + classes + 1)
         + entry * w * h + loc;
}

static int max_index(float* a, int n) {
  if (n <= 0) return -1;
  int i, max_i = 0;
  float max = a[0];
  for (i = 1; i < n; ++i) {
    if (a[i] > max) {
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}

static std::vector<yolov3_DetectRect> detection_yolov3_process(
    detection*    dets,
    const st_process_info& proc_info,
    int           nboxes,
    int           cols,
    int           rows) {
  int classes = proc_info.classes_num_;
  float thresh = proc_info.threshold_prob_;
  std::vector<yolov3_DetectRect> dets_all;
  for (int k = 0; k < nboxes; k++) {
    yolov3_box b = dets[k].bbox;
    int left = (b.x - b.w / 2.) * cols;
    int right = (b.x + b.w / 2.) * cols;
    int top = (b.y - b.h / 2.) * rows;
    int bot = (b.y + b.h / 2.) * rows;
    bot = std::max(bot, top);
    top = std::min(bot, top);
    if (left < 0) left = 0;
    if (right > cols - 1) right = cols - 1;
    if (top < 0) top = 0;
    if (bot > rows - 1) bot = rows - 1;
    int category = max_index(dets[k].prob, classes);
    if (dets[k].prob[category] > thresh) {
      yolov3_DetectRect det_k;
      det_k.left = left;
      det_k.right = right;
      det_k.top = top;
      det_k.bot = bot;
      det_k.category = category;
      det_k.score = dets[k].prob[category];
      dets_all.push_back(det_k);
    }
  }
  return dets_all;
}

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
         char* obj_num) {

  st_process_info proc_info;
  proc_info.classes_num_ = class_num;
  proc_info.threshold_prob_ = thresh_hold;
  proc_info.anchor_num_ = 3;
  proc_info.threshold_nms_ = 0.45;

  int* net_shape_int = (int*)net_shape;
  proc_info.net_w_ = net_shape_int[3];
  proc_info.net_h_ = net_shape_int[2];
  proc_info.biases_ = (float*)anchor_biases;
  proc_info.biases_num_ = anchor_biases_num;
  proc_info.masks_ = (int*)masks;
  int* input_shape_ptr = (int*)input_shape;
  std::vector<int> tensor_sizes;
  for (size_t i = 0; i < input_tensor_num; i++) {
    int length = 1;
    for (size_t j = 1; j < 4; j++) {
      length *= input_shape_ptr[i * 4 + j];
    }
    tensor_sizes.push_back(length);
    proc_info.fm_size_[i * 2] = input_shape_ptr[i * 4 + 3];
    proc_info.fm_size_[i * 2 + 1] = input_shape_ptr[i * 4 + 2];
  }
  std::vector<float*> f_inputs;
  for(size_t i = 0; i < input_tensor_num; i++) {
    long ptr = (long)inputs[i];
    f_inputs.push_back((float*)ptr);
  }
  int batch_size =  net_shape_int[0];
  float* out_results = (float*)results;
  int* i_obj_num = (int*)obj_num;
  *i_obj_num = 0;
  for (size_t i = 0; i < batch_size; i++) {
    int nboxes = 0;
    std::vector<float*> blobs;
    for(size_t j = 0; j < input_tensor_num; j++) {
      float* tmp_ptr = (float*)f_inputs[j];
      blobs.push_back(tmp_ptr + tensor_sizes[j] * i);
    }
    detection* dets = get_detections(blobs, proc_info,
                img_width, img_height, &nboxes);
    std::vector<yolov3_DetectRect> det_result =
                   detection_yolov3_process(dets, proc_info, nboxes,
                         img_width, img_height);
    free_detections(dets, nboxes);
    for(size_t j = 0; j < det_result.size(); j++) {
      out_results[*i_obj_num * 7] = i;
      out_results[*i_obj_num * 7 + 1] = det_result[j].category;
      out_results[*i_obj_num * 7 + 2] = det_result[j].score;
      out_results[*i_obj_num * 7 + 3] = det_result[j].left;
      out_results[*i_obj_num * 7 + 4] = det_result[j].top;
      out_results[*i_obj_num * 7 + 5] = det_result[j].right;
      out_results[*i_obj_num * 7 + 6] = det_result[j].bot;
       *i_obj_num += 1;
      if (*i_obj_num >= top_k) {
        return;
      }
    }
  }
}
