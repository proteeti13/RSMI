#ifndef MODELTOOLS_H
#define MODELTOOLS_H

// Set to false before calling build() to silence per-node training logs.
inline bool rsmi_verbose = true;

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <xmmintrin.h>

using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace std;

struct Net : torch::nn::Module
{

public:
    int input_width;
    int max_error = 0;
    int min_error = 0;
    int width = 0;

    float learning_rate = Constants::LEARNING_RATE;

    float w1[Constants::HIDDEN_LAYER_WIDTH * 3];   // 3 inputs x hidden
    float w1_[Constants::HIDDEN_LAYER_WIDTH];
    float w2[Constants::HIDDEN_LAYER_WIDTH];
    float b1[Constants::HIDDEN_LAYER_WIDTH];

    float *w1_0 = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);  // SourceID weights
    float *w1_1 = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);  // Hop1_ID weights
    float *w1_2 = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);  // Hop2_ID weights
    float *w2_  = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *b1_  = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);

    float *w1__ = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);

    float b2 = 0.0;

    Net(int input_width)
    {
        this->width = Constants::HIDDEN_LAYER_WIDTH;
        this->input_width = input_width;
        fc1 = register_module("fc1", torch::nn::Linear(input_width, width));
        fc2 = register_module("fc2", torch::nn::Linear(width, 1));
        torch::nn::init::uniform_(fc1->weight, 0, 1);
        torch::nn::init::uniform_(fc2->weight, 0, 1);
    }

    // RSMI
    Net(int input_width, int width)
    {
        this->width = width;
        this->width = this->width >= Constants::HIDDEN_LAYER_WIDTH ? Constants::HIDDEN_LAYER_WIDTH : this->width;
        this->input_width = input_width;
        fc1 = register_module("fc1", torch::nn::Linear(input_width, this->width));
        fc2 = register_module("fc2", torch::nn::Linear(this->width, 1));
        torch::nn::init::uniform_(fc1->weight, 0, 0.1);
        torch::nn::init::uniform_(fc2->weight, 0, 0.1);
    }

    void get_parameters_ZM()
    {
        torch::Tensor p1 = this->parameters()[0];
        torch::Tensor p2 = this->parameters()[1];
        torch::Tensor p3 = this->parameters()[2];
        torch::Tensor p4 = this->parameters()[3];
        p1 = p1.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            w1_[i] = p1.select(0, i).item().toFloat();
        }

        p2 = p2.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            b1[i] = p2.select(0, i).item().toFloat();
        }

        p3 = p3.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            w2[i] = p3.select(0, i).item().toFloat();
        }
        b2 = p4.item().toFloat();
    }

    // Extract fc1 weights into SSE-aligned arrays for 3-input fast inference.
    // PyTorch stores fc1->weight as [hidden_width, input_width] row-major,
    // so neuron i has weights at positions: 3*i (SourceID), 3*i+1 (Hop1_ID), 3*i+2 (Hop2_ID).
    void get_parameters()
    {
        torch::Tensor p1 = this->parameters()[0];
        torch::Tensor p2 = this->parameters()[1];
        torch::Tensor p3 = this->parameters()[2];
        torch::Tensor p4 = this->parameters()[3];
        p1 = p1.reshape({3 * width, 1});
        for (size_t i = 0; i < (size_t)width; i++)
        {
            w1[i * 3]     = p1.select(0, 3 * i).item().toFloat();
            w1[i * 3 + 1] = p1.select(0, 3 * i + 1).item().toFloat();
            w1[i * 3 + 2] = p1.select(0, 3 * i + 2).item().toFloat();

            w1_0[i] = p1.select(0, 3 * i).item().toFloat();
            w1_1[i] = p1.select(0, 3 * i + 1).item().toFloat();
            w1_2[i] = p1.select(0, 3 * i + 2).item().toFloat();
        }

        p2 = p2.reshape({width, 1});
        for (size_t i = 0; i < (size_t)width; i++)
        {
            b1[i]  = p2.select(0, i).item().toFloat();
            b1_[i] = p2.select(0, i).item().toFloat();
        }

        p3 = p3.reshape({width, 1});
        for (size_t i = 0; i < (size_t)width; i++)
        {
            w2[i]  = p3.select(0, i).item().toFloat();
            w2_[i] = p3.select(0, i).item().toFloat();
        }
        b2 = p4.item().toFloat();
    }

    void print_parameters()
    {
        for (size_t i = 0; i < (size_t)width; i++)
        {
            cout << b1[i] << " " << b1[i] << " ";
        }
        cout << endl;
        for (size_t i = 0; i < (size_t)width; i++)
        {
            cout << b1_[i] << " " << b1_[i] << " ";
        }
        cout << endl;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::Tensor predict(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    float predict_ZM(float key)
    {
        int blocks = width / 4;
        int rem = width % 4;
        int move_back = blocks * 4;
        __m128 fLoad_w1, fLoad_b1, fLoad_w2;
        __m128 temp1, temp2, temp3;
        __m128 fSum0 = _mm_setzero_ps();
        __m128 fLoad0_x, fLoad0_zeros;
        fLoad0_x = _mm_set_ps(key, key, key, key);
        fLoad0_zeros = _mm_set_ps(0, 0, 0, 0);
        float result;
        for (int i = 0; i < blocks; i++)
        {
            fLoad_w1 = _mm_load_ps(w1__);
            fLoad_b1 = _mm_load_ps(b1_);
            fLoad_w2 = _mm_load_ps(w2_);
            temp1 = _mm_mul_ps(fLoad0_x, fLoad_w1);
            temp2 = _mm_add_ps(temp1, fLoad_b1);
            temp1 = _mm_max_ps(temp2, fLoad0_zeros);
            temp2 = _mm_mul_ps(temp1, fLoad_w2);
            fSum0 = _mm_add_ps(fSum0, temp2);
            w1__ += 4;
            b1_  += 4;
            w2_  += 4;
        }
        result = 0;
        if (blocks > 0)
        {
            result += fSum0[0] + fSum0[1] + fSum0[2] + fSum0[3];
        }
        for (size_t i = 0; i < (size_t)rem; i++)
        {
            result += activation(key * w1__[i] + b1_[i]) * w2_[i];
        }
        result += b2;
        w1__ -= move_back;
        b1_  -= move_back;
        w2_  -= move_back;
        return result;
    }

    // SSE-accelerated 3-input inference: relu(x*w1_0 + y*w1_1 + z*w1_2 + b1) . w2 + b2
    float predict(Point point)
    {
        float x1 = point.x;
        float x2 = point.y;
        float x3 = point.z;
        int blocks = width / 4;
        int rem = width % 4;
        int move_back = blocks * 4;
        __m128 fLoad_w1_0, fLoad_w1_1, fLoad_w1_2, fLoad_b1, fLoad_w2;
        __m128 temp1, temp2, temp3;
        __m128 fSum0 = _mm_setzero_ps();
        __m128 fLoad0_x1, fLoad0_x2, fLoad0_x3, fLoad0_zeros;
        fLoad0_x1    = _mm_set_ps(x1, x1, x1, x1);
        fLoad0_x2    = _mm_set_ps(x2, x2, x2, x2);
        fLoad0_x3    = _mm_set_ps(x3, x3, x3, x3);
        fLoad0_zeros = _mm_set_ps(0, 0, 0, 0);
        float result;
        for (int i = 0; i < blocks; i++)
        {
            fLoad_w1_0 = _mm_load_ps(w1_0);
            fLoad_w1_1 = _mm_load_ps(w1_1);
            fLoad_w1_2 = _mm_load_ps(w1_2);
            fLoad_b1   = _mm_load_ps(b1_);
            fLoad_w2   = _mm_load_ps(w2_);

            temp1 = _mm_mul_ps(fLoad0_x1, fLoad_w1_0);   // x * w0
            temp2 = _mm_mul_ps(fLoad0_x2, fLoad_w1_1);   // y * w1
            temp3 = _mm_mul_ps(fLoad0_x3, fLoad_w1_2);   // z * w2

            temp1 = _mm_add_ps(temp1, temp2);
            temp1 = _mm_add_ps(temp1, temp3);
            temp1 = _mm_add_ps(temp1, fLoad_b1);          // + bias
            temp1 = _mm_max_ps(temp1, fLoad0_zeros);      // ReLU

            temp2 = _mm_mul_ps(temp1, fLoad_w2);
            fSum0 = _mm_add_ps(fSum0, temp2);

            w1_0 += 4;
            w1_1 += 4;
            w1_2 += 4;
            b1_  += 4;
            w2_  += 4;
        }
        result = 0;
        if (blocks > 0)
        {
            result += fSum0[0] + fSum0[1] + fSum0[2] + fSum0[3];
        }
        for (size_t i = 0; i < (size_t)rem; i++)
        {
            result += activation(x1 * w1_0[i] + x2 * w1_1[i] + x3 * w1_2[i] + b1_[i]) * w2_[i];
        }
        result += b2;
        w1_0 -= move_back;
        w1_1 -= move_back;
        w1_2 -= move_back;
        b1_  -= move_back;
        w2_  -= move_back;
        return result;
    }

    float activation(float val)
    {
        if (val > 0.0)
        {
            return val;
        }
        return 0.0;
    }

    void train_model(vector<float> locations, vector<float> labels)
    {
        long long N = labels.size();

#ifdef use_gpu
        torch::Tensor x = torch::tensor(locations, at::kCUDA).reshape({N, this->input_width});
        torch::Tensor y = torch::tensor(labels, at::kCUDA).reshape({N, 1});
#else
        torch::Tensor x = torch::tensor(locations).reshape({N, this->input_width});
        torch::Tensor y = torch::tensor(labels).reshape({N, 1});
#endif
        if (rsmi_verbose) cout << "trained size: " << N << endl;

        torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(this->learning_rate));
        if (N > 64000000)
        {
            int batch_num = 4;

            auto x_chunks = x.chunk(batch_num, 0);
            auto y_chunks = y.chunk(batch_num, 0);
            for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
            {
                for (size_t i = 0; i < batch_num; i++)
                {
                    optimizer.zero_grad();
                    torch::Tensor loss = torch::mse_loss(this->forward(x_chunks[i]), y_chunks[i]);
#ifdef use_gpu
                    loss.to(torch::kCUDA);
#endif
                    loss.backward();
                    optimizer.step();
                }
            }
        }
        else
        {
            for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
            {
                optimizer.zero_grad();
                torch::Tensor loss = torch::mse_loss(this->forward(x), y);
#ifdef use_gpu
                loss.to(torch::kCUDA);
#endif
                loss.backward();
                optimizer.step();
            }
        }
        if (rsmi_verbose) cout << "finish training " << endl;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

#endif
