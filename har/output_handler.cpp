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

#include "output_handler.h"
#include "Arduino.h"



void HandleOutput( int kind) {
  // The first time this method runs, set up our LED
  static bool is_initialized = false;
  // Toggle the LED every time an inference is performed
  static int count = 0;
  ++count;

  // Print some ASCII art for each gesture
  if (kind == 0) {
    // error_reporter->Report("\n\r█ Wingardium Leviosa █\n\r");
    Serial.println("Downstairs\n\r");
  } else if (kind == 1) {
    // error_reporter->Report("\n\r█ Obliviate █\n\r");
    Serial.println("Jogging\n\r");
  } else if (kind == 2) {
    // error_reporter->Report("\n\r█ Lumos █\n\r");
    Serial.println("Sitting\n\r");
  } else if (kind == 3) {
    // error_reporter->Report("\n\r█ Lumos █\n\r");
    Serial.println("Standing\n\r");
  } else if (kind == 4) {
    // error_reporter->Report("\n\r█ Lumos █\n\r");
    Serial.println("Upstairs\n\r");
  } else if (kind == 5) {
    // error_reporter->Report("\n\r█ Lumos █\n\r");
    Serial.println("Walking\n\r");
  }
}
