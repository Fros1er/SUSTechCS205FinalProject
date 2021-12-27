#include <iostream>
#include <chrono>
#include <functional>
#include <fstream>
#include "Matrix.hpp"
#include "opencv2/opencv.hpp"
#include "face_binary_cls.hpp"
#include "cnn.hpp"

#pragma GCC optimize(3, "Ofast", "inline")

template<class T>
T counter(double &time, std::function<T()> func) {
    auto start = std::chrono::steady_clock::now();
    T res = func();
    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration<double, std::milli>(end - start).count();
    return res;
}

int main() {
//    std::ifstream s("../matrices/mat-A-4.txt");
//    Matrix<float> inn(4, 4, 1);
//    s >> inn;
//    std:: cout << convBnReLU(inn, std::vector<float>{1, 2, 3, 4, 5 ,6 ,7 ,8, 9}, std::vector<float>{0}, 3, 1, 2, 1);
//
//    return 0;

    cv::Mat cv_in_t = cv::imread("../images/trump.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    cv::Mat cv_in;
    cv::resize(cv_in_t, cv_in, {128, 128});
    Matrix<float> in(128, 128, 3);
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            cv::Vec3b p = cv_in.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; c++) {
                in(i, j, c) = static_cast<float>(p[c]) / 255.0f;
            }
        }
    }

    std::vector<float> weight{conv0_weight, std::end(conv0_weight)};
    std::vector<float> bias{conv0_bias, std::end(conv0_bias)};

    double time = 0.0, total = 0.0;
    auto out = counter<Matrix<float>>(time, [&](){
        return convBnReLU(in, weight, bias, 3, 16, 2, 1);
    });
    total += time;
    std::cout << "Conv 1: " << time << "ms" << std::endl;
    out = counter<Matrix<float>>(time, [&](){
        return maxPool2D(out, 2);
    });
    total += time;
    std::cout << "pool 1: " << time << "ms" << std::endl;

    weight = {conv1_weight, std::end(conv1_weight)};
    bias = {conv1_bias, std::end(conv1_bias)};
    out = counter<Matrix<float>>(time, [&](){
        return convBnReLU(out, weight, bias, 3, 32, 1, 0);
    });
    total += time;
    std::cout << "Conv 2: " << time << "ms" << std::endl;
    out = counter<Matrix<float>>(time, [&](){
        return maxPool2D(out, 2);
    });
    total += time;
    std::cout << "pool 2: " << time << "ms" << std::endl;

    weight = {conv2_weight, std::end(conv2_weight)};
    bias = {conv2_bias, std::end(conv2_bias)};
    out = counter<Matrix<float>>(time, [&](){
        return convBnReLU(out, weight, bias, 3, 32, 2, 1);
    });
    total += time;
    std::cout << "Conv 3: " << time << "ms" << std::endl;

    auto flattened = counter<std::vector<float>>(time, [&](){
        return out.flat();
    });
    total += time;
    std::cout << "flat: " << time << "ms" << std::endl;

    weight = {fc0_weight, std::end(fc0_weight)};
    bias = {fc0_bias, std::end(fc0_bias)};
    auto res = counter<std::vector<float>>(time, [&](){
        return fullConnect(flattened, weight, bias);
    });
    total += time;
    std::cout << "fc: " << time << "ms" << std::endl;
    res = softMax(res);

    std::cout << res[0] << " " << res[1] << std::endl;
    std::cout << "total_cal_time: " << total << "ms" << std::endl;

    return 0;
}
