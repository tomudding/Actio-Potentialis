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

MLPNetwork::MLPNetwork(std::vector<int> const _layers) {
    this->number_of_layers = _layers.size() - 1;
    
    for (int i = 0; i < this->number_of_layers; i++) {
        layers.emplace_back(_layers[i], _layers[i + 1]);
    }
}

MLPNetwork::MLPNetwork(std::vector<int> const _layers, std::vector<float> _biases, std::vector<std::vector<std::vector<float>>> _weights) {
    this->number_of_layers = _layers.size() - 1;
    
    for (int i = 0; i < this->number_of_layers; i++) {
        layers.emplace_back(_layers[i], _layers[i + 1], _biases[i], _weights[i]);
    }
}

void MLPNetwork::learn(std::vector<float> _input, std::vector<float> _expected, int _epochs) {
    // for 'epochs' feed the network input and propagate the expected
    // output back through the network
    for (int i = 0; i < _epochs; i++) {
        this->feed_propagate(_input);
        this->propagate(_expected);
    }
}

void MLPNetwork::learn(std::vector<float> _input, std::vector<float> _expected) {
    this->learn(_input, _expected, 1);
}

void MLPNetwork::learn(std::vector<std::vector<float>> _input, std::vector<std::vector<float>> _expected, int _epochs) {
    // for 'epochs' feed the network input and propagate the expected
    // output back through the network
    for (int i = 0; i < _epochs; i++) {
        std::vector<std::vector<float>>::iterator it;
        int j = 0;
        
        for (it = _input.begin(); it != _input.end(); it++, j++) {
            this->feed_propagate(_input[j]);
            this->propagate(_expected[j]);
        }
    }
}

void MLPNetwork::learn(std::vector<std::vector<float>> _input, std::vector<std::vector<float>> _expected) {
    this->learn(_input, _expected, 1);
}

std::vector<float> MLPNetwork::classify(std::vector<float> _input) {
    // feed the network input and return the result
    return this->feed_classify(_input);
}

std::vector<std::vector<float>> MLPNetwork::classify(std::vector<std::vector<float>> _input) {
    std::vector<std::vector<float>> output;
    std::vector<std::vector<float>>::iterator it;
    int j = 0;
    
    for (it = _input.begin(); it != _input.end(); it++, j++) {
        output.push_back(this->classify(_input[j]));
    }
    
    return output;
}

void MLPNetwork::output() {
    Serial.println(F("std::vector<std::vector<std::vector<float> weights{"));
    for (auto &layer : layers) {
        Serial.println(F("    {"));
        for (auto &in_neuron : layer.current_weights) {
            Serial.print(F("        {"));
            for (auto &value : in_neuron) {
                Serial.print(value, 7); Serial.print(F(", "));
            }
            Serial.println(F("},"));
        }
        Serial.print(F("},"));
    }
    Serial.println(F("};"));
    
    Serial.println(F("std::vector<float> biases{"));
    for (auto &layer: layers) {
        Serial.print(layer.current_bias, 7); Serial.println(F(","));
    }
    Serial.println(F("};"));
}

void MLPNetwork::feed_propagate(std::vector<float> _input) {
    _inputs = _input;
    
    // feed the input layer and forward (read calculate output)
    layers[0].forward(_inputs);
    
    // forward through all hidden layers
    for (int i = 1; i < this->number_of_layers; i++) {
        layers[i].forward(layers[i - 1].current_output);
    }
}

std::vector<float> MLPNetwork::feed_classify(std::vector<float> _input) {
    _inputs = _input;
    
    // feed the input layer and forward (read calculate output)
    layers[0].forward(_inputs);
    
    // forward through all hidden layers
    for (int i = 1; i < this->number_of_layers; i++) {
        layers[i].forward(layers[i - 1].current_output);
    }
    
    // return output of output layer
    return layers[this->number_of_layers - 1].current_output;
}

void MLPNetwork::propagate(std::vector<float> _expected) {
    // start backpropagation through all layers
    layers[this->number_of_layers - 1].propagate_output_layer(layers[this->number_of_layers - 2].current_output, _expected);

    // propagate back through all hidden layers
    for (int i = this->number_of_layers - 2; i > 0; i--) {
        layers[i].propagate_hidden_layer(layers[i - 1].current_output, layers[i + 1]);
    }
        
    // propagate back to the input layer
    layers[0].propagate_hidden_layer(_inputs, layers[1]);
}