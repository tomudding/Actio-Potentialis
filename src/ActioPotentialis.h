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
#ifndef ACTIOPOTENTIALIS_ACTIOPOTENTIALIS_H_
#define ACTIOPOTENTIALIS_ACTIOPOTENTIALIS_H_

#include <vector>

class MLPLayer {
    public:
        std::vector<std::vector<float>> current_weights;    // up-to-date weights of the current layer
        std::vector<float> current_output;                  // output of the output neurons in the current layer
        std::vector<float> previous_error;                  // error of the output neurons in the current layer
        float current_bias;                                 // bias of the current layer
        
        unsigned int _numberOfInputs;
        unsigned int _numberOfOutputs;
        
        // Constructors
        MLPLayer(unsigned int _n_inputs, unsigned int _n_outputs);
        MLPLayer(unsigned int _n_inputs, unsigned int _n_outputs, float _default_bias, std::vector<std::vector<float>> _default_weights);
        
        // Functions
        void forward(std::vector<float> _inputs);
        void propagate_hidden_layer(std::vector<float> _input, MLPLayer _next_layer);
        void propagate_output_layer(std::vector<float> _input, std::vector<float> _expected);
        
        // Activation function(s)
        float sigmoid(float _x);
        float sigmoid_der(float _x);
    private:
        // Learning rates
        const float _learning_rate_weights = 0.33;
        const float _learning_rate_bias = 0.66;
};

class MLPNetwork {
    public:
        // Pointers / Variables
        std::vector<MLPLayer> layers;
        
        // Constructors
        MLPNetwork(std::vector<int> _layers);
        MLPNetwork(std::vector<int> _layers, std::vector<float> _biases, std::vector<std::vector<std::vector<float>>> _weights);
        
        // Functions
        void learn(std::vector<float> _input, std::vector<float> _expected, int _epochs);
        void learn(std::vector<float> _input, std::vector<float> _expected);
        void learn(std::vector<std::vector<float>> _input, std::vector<std::vector<float>> _expected, int _epochs);
        void learn(std::vector<std::vector<float>> _input, std::vector<std::vector<float>> _expected);
        std::vector<float> classify(std::vector<float> _input);
        std::vector<std::vector<float>> classify(std::vector<std::vector<float>> _input);
        
        void output();
    private:        
        // Pointers / Variables
        std::vector<float> _inputs;
        unsigned int number_of_layers;
        
        // Functions
        void feed_propagate(std::vector<float> _input);
        std::vector<float> feed_classify(std::vector<float> _input);
        void propagate(std::vector<float> _expected);
};

#endif // ACTIOPOTENTIALIS_ACTIOPOTENTIALIS_H_