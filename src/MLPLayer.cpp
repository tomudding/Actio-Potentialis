/*!
 * This file is part of the 'Actio Potentialis' library.
 * Copyright (C) 2019 Tom Udding. All rights reserved.
 * Copyright (c) 2019 George Chousos. All rights reserved.
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
#include "ActioPotentialis.h"
#include "Arduino.h"

#include <vector>

MLPLayer::MLPLayer(unsigned int _n_inputs, unsigned int _n_outputs) {
    _numberOfInputs = _n_inputs;
    _numberOfOutputs = _n_outputs;
    
    current_output = std::vector<float>(_numberOfOutputs);
    current_bias = 1.0f;
    
    // generate "random" weights between the neurons
    for (int j = 0; j < _numberOfInputs; j++) {
        current_weights.push_back(std::vector<float>(_numberOfOutputs));
        
        for (int i = 0; i < _numberOfOutputs; i++) {
            current_weights[j][i] = random(-10000, 10000) / 10000.0;
        }
    }
}

MLPLayer::MLPLayer(unsigned int _n_inputs, unsigned int _n_outputs, float _default_bias, std::vector<std::vector<float>> _default_weights) {
    _numberOfInputs = _n_inputs;
    _numberOfOutputs = _n_outputs;
    
    current_output = std::vector<float>(_numberOfOutputs);
    current_bias = _default_bias;
    current_weights = _default_weights;
}

void MLPLayer::forward(std::vector<float> _inputs) {
    // for all neurons in the next layer
    for (int i = 0; i < _numberOfOutputs; i++) {
        current_output[i] = 0.0f;
        
        // sum from previous layer
        for (int j = 0; j < _numberOfInputs; j++) {
            current_output[i] += _inputs[j] * current_weights[j][i];
        }
        
        // calculate output of the layer using the activation function
        current_output[i] = this->sigmoid(current_output[i] + current_bias);
    }
}

void MLPLayer::propagate_hidden_layer(std::vector<float> _input, MLPLayer _next_layer) {
    // backpropagation, thus create errors backwards
    previous_error = std::vector<float>(_numberOfInputs);
    float error_delta;
    float bias_delta = 1.0f;
        
    // for all outputs calculate the error based on the expected output
    for (int i = 0; i < _numberOfOutputs; i++) {
        // error is derived from the error of the previous (next) layer
        error_delta = _next_layer.previous_error[i] * this->sigmoid_der(current_output[i]);
        bias_delta *= error_delta;
        
        // do actual backpropagation and update weights based on the
        // calculated errors
        for (int j = 0; j < _numberOfInputs; j++) {
            previous_error[j] += error_delta * current_weights[j][i];
            current_weights[j][i] -= error_delta * _input[j] * _learning_rate_weights;
        }
    }
    
    current_bias -= bias_delta * _learning_rate_bias;
}

void MLPLayer::propagate_output_layer(std::vector<float> _input, std::vector<float> _expected) {
    // backpropagation, thus create errors backwards
    previous_error = std::vector<float>(_numberOfInputs);
    float error_delta;
    float bias_delta = 1.0f;

    // for all outputs calculate the error based on the expected output
    for (int i = 0; i < _numberOfOutputs; i++) {
        // error is the difference of the expected output and the
        // actual output provided by the network
        error_delta = ((2 / _numberOfOutputs) * (current_output[i] - _expected[i]) * this->sigmoid_der(current_output[i]));
        bias_delta *= error_delta;
        
        // do actual backpropagation and update weights based on the
        // calculated errors
        for (int j = 0; j < _numberOfInputs; j++) {
            previous_error[j] += error_delta * current_weights[j][i];
            current_weights[j][i] -= error_delta * _input[j] * _learning_rate_weights;
        }
    }
    
    current_bias -= bias_delta * _learning_rate_bias;
}

float MLPLayer::sigmoid(float _x) {
    return 1 / (1 + exp(-_x));
}

float MLPLayer::sigmoid_der(float _x) {
    return _x * (1 - _x);
}