/* 
Adapted by Andri Yadi.
Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "accelerometer_handler.h"
#include <Arduino.h>
#include "Arduino_BMI270_BMM150.h"

#include "constants.h"

#define RING_BUFFER_SIZE  600

// A buffer holding the last 200 sets of 3-channel values
float save_data[RING_BUFFER_SIZE] = {0.0};
// Most recent position in the save_data buffer
int begin_index = 0;
// True if there is not yet enough data to run inference
bool pending_initial_data = true;
// How often we should save a measurement during downsampling
int sample_every_n;
// The number of measurements since we last saved one
int sample_skip_counter = 1;

TfLiteStatus SetupAccelerometer() {
  
  // Switch on the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU");
    return kTfLiteError;
  }

  // Determine how many measurements to keep in order to meet kTargetHz
  float sample_rate = IMU.accelerationSampleRate();
  sample_every_n = static_cast<int>(roundf(sample_rate / kTargetHz));

  return kTfLiteOk;
}

bool ReadAccelerometer(int8_t* input,
                       int input_length, bool reset_buffer,float input_scale ,float input_zero_point) {
  // Clear the buffer if required, e.g. after a successful prediction
  if (reset_buffer) {
    memset(save_data, 0, RING_BUFFER_SIZE * sizeof(float));
    begin_index = 0;
    pending_initial_data = true;
  }
  // Keep track of whether we stored any new data
  bool new_data = false;
  // Loop through new samples and add to buffer
  while (IMU.accelerationAvailable()) {
    float x, y, z;
    // Read each sample, removing it from the device's FIFO buffer
    if (!IMU.readAcceleration(x, y, z)) {
      Serial.println("Failed to read data");
      break;
    }
    // Throw away this sample unless it's the nth
    if (sample_skip_counter != sample_every_n) {
      sample_skip_counter += 1;
      continue;
    }

    // Write samples to our buffer, converting to milli-Gs
    // Change board orientation (for my purpose) that's specific for 
    // Arduino Nano BLE Sense, for compatibility with model 
    // (sensor orientation is different on Arduino Nano BLE Sense 
    // compared with SparkFun Edge)
    save_data[begin_index++] = y*10;
    save_data[begin_index++] = x*10;
    save_data[begin_index++] = z*10;
    // Since we took a sample, reset the skip counter
    sample_skip_counter = 1;
    // If we reached the end of the circle buffer, reset
    if (begin_index >= RING_BUFFER_SIZE) {
      begin_index = 0;
    }
    new_data = true;
  }

  // Skip this round if data is not ready yet
  if (!new_data) {
    return false;
  }

  // Check if we are ready for prediction or still pending more initial data
  if (pending_initial_data && begin_index >= 200) {
    pending_initial_data = false;
  }

  // Return if we don't have enough data
  if (pending_initial_data) {
    return false;
  }

  // Copy the requested number of bytes to the provided input tensor
  for (int i = 0; i < input_length; ++i) {
    int ring_array_index = begin_index + i - input_length;
    if (ring_array_index < 0) {
      ring_array_index += 600;
    }
    input[i] = static_cast<int8_t>(save_data[ring_array_index]/input_scale+input_zero_point);
    //input[i] = save_data[ring_array_index];
  }

  return true;
}
