/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>

#include "accelerometer_handler.h"
#include "gesture_predictor.h"
#include "magic_wand_model_data.h"
#include "output_handler.h"
#include <cmath>
#include <chrono>
#include "magic_wand_model_data.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define BLE_SENSE_UUID(val) ("4798e0f2-" val "-4d68-af64-8a8f5258404e")

#undef MAGIC_WAND_DEBUG
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16)uint8_t tensor_arena[kTensorArenaSize];
float input_scale;
int input_zero_point;

// 获取输出张量的量化参数
float output_scale;
int output_zero_point;

// Whether we should clear the buffer next time we fetch data
bool should_clear_buffer = false;
}  // namespace

void setup()
{
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.


  // Wait until we know the serial port is ready
  while (!Serial) {
  }

  delay(500);
  MicroPrintf("\n\r\n\rMagic Wand - TensorFlow Lite demo");
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
      MicroPrintf("TFLite Model provided is schema version %d, which not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  else {
    MicroPrintf("TFLite Model provided is schema version %d.", model->version());    
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //static tflite::ops::micro::AllOpsResolver micro_mutable_op_resolver;

  static tflite::MicroMutableOpResolver<6> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddUnidirectionalSequenceLSTM();
  //micro_op_resolver.AddStridedSlice();
  /*micro_op_resolver.AddBatchToSpaceNd();
  micro_op_resolver.AddSpaceToBatchNd();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddQuantize();
  /*micro_op_resolver.AddPack();
  micro_op_resolver.AddFill();
  micro_op_resolver.AddWhile();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddLess();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddGather();
  micro_op_resolver.AddSplit();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddTanh();
  micro_op_resolver.AddSlice();*/
  //static tflite::AllOpsResolver micro_op_resolver;
  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors
  
  interpreter->AllocateTensors();
  // Obtain pointer to the model's input tensor
  model_input = interpreter->input(0);
    // 获取输入张量的量化参数
  Serial.println(model_input->dims->data[0]);
  Serial.println(model_input->dims->data[1]);
  Serial.println(model_input->dims->data[2]);
  Serial.println(model_input->dims->size);
  Serial.println(model_input->type);


  


  input_scale = model_input->params.scale;
  input_zero_point = model_input->params.zero_point;

  // 获取输出张量的量化参数
  output_scale = interpreter->output(0)->params.scale;
  output_zero_point = interpreter->output(0)->params.zero_point;
  
  //input_length = model_input->bytes / sizeof(float);
  
  input_length = model_input->bytes / sizeof(int);
  TfLiteStatus setup_status = SetupAccelerometer();
  if (setup_status != kTfLiteOk) {
    MicroPrintf("Set up failed\n");
  }
  else {
    MicroPrintf("Magic starts!\n");
    MicroPrintf("\r\nPredicted gestures:\n\r");
  }
}

void loop()
{
  // Attempt to read new data from the accelerometer
  bool got_data = ReadAccelerometer(model_input->data.int8,
                                    input_length, should_clear_buffer,input_scale ,input_zero_point);
  //bool got_data = ReadAccelerometer(model_input->data.f,
//                                    input_length, should_clear_buffer,input_scale ,input_zero_point);
  if (should_clear_buffer) {
    MicroPrintf("\r\nPredicted gestures:\n\r");
  }

  // Don't try to clear the buffer again
  should_clear_buffer = false;
  // If there was no new data, wait until next time
  if (!got_data) {
    return;
  }
  auto start = std::chrono::high_resolution_clock::now();
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  MicroPrintf("%f\n", elapsed);
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on index: %d\n", begin_index);
    return;
  }
  // Analyze the results to obtain a prediction
  int gesture_index = PredictGesture(interpreter->output(0)->data.int8,output_scale,output_zero_point);
  //int gesture_index = PredictGesture(interpreter->output(0)->data.f,output_scale,output_zero_point);
  // Clear the buffer next time we read data
  should_clear_buffer = gesture_index < 6;
  // Produce an output
  HandleOutput(gesture_index);
}

