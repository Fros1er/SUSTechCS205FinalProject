#pragma once

#include "Matrix.hpp"

template<typename T, typename U, typename V>
Matrix<T>
convBnReLU(const Matrix<T> &target, const std::vector<U> &kernels, const std::vector<V> &bias, const size_t &kernel_size,
           const size_t &outChannel, const size_t &stride = 1, const size_t &padding = 0) {
#ifndef NDEBUG
    if (outChannel * target.getNumberOfChannels() * kernel_size * kernel_size != kernels.size()) throw std::invalid_argument("Bad kernels");
    if (bias.size() != outChannel) throw std::invalid_argument("Bad bias size");
        if (stride < 1) throw std::invalid_argument("Bad stride");
#endif
    Matrix<T> res(1 + (target.getNumberOfRows() + padding * 2 - kernel_size) / stride,
                  1 + (target.getNumberOfCols() + padding * 2 - kernel_size) / stride,
                  outChannel);

//#pragma omp parallel for
    for (size_t out = 0; out < outChannel; out++) {
        for (size_t in = 0; in < target.getNumberOfChannels(); in++) {
            std::vector<U> kernel(kernel_size * kernel_size);
            for (size_t k = 0; k < kernel_size * kernel_size; k++) {
                kernel[k] = kernels[out * target.getNumberOfChannels() * kernel_size * kernel_size +
                                    in * kernel_size * kernel_size + k];
            }
            for (size_t i = 0; i < 1 + target.getNumberOfCols() - kernel_size + padding * 2; i += stride) {
                for (size_t j = 0; j < 1 + target.getNumberOfRows() - kernel_size + padding * 2; j += stride) {
                    for (size_t k = 0; k < kernel_size; k++) {
                        for (size_t l = 0; l < kernel_size; l++) {
                            int64_t tmpI = i + k, tmpJ = j + l;
                            tmpI -= padding;
                            tmpJ -= padding;
                            T num = (tmpI < 0 || tmpI >= target.getNumberOfRows() || tmpJ < 0 ||
                                     tmpJ >= target.getNumberOfCols()) ? 0 : target(tmpI, tmpJ, in);
                            res(i / stride, j / stride, out) += num * kernel[k * kernel_size + l];
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < res.getNumberOfRows(); i++) {
            for (size_t j = 0; j < res.getNumberOfCols(); j++) {
                res(i, j, out) = std::max(T(), res(i, j, out) + bias[out]);
            }
        }
    }
    return res;
}

template<typename T>
Matrix<T> maxPool2D(const Matrix<T> &target, const size_t &size) {
#ifndef NDEBUG
    if (target.getNumberOfRows() % size != 0 || target.getNumberOfCols() % size != 0) throw std::invalid_argument("bad pool args");
#endif
    Matrix<T> res(target.getNumberOfRows() / size, target.getNumberOfCols() / size, target.getNumberOfChannels());
//#pragma omp parallel for
    for (size_t c = 0; c < target.getNumberOfChannels(); c++) {
        for (size_t i = 0; i < target.getNumberOfRows(); i += size) {
            for (size_t j = 0; j < target.getNumberOfCols(); j += size) {
                for (size_t m = 0; m < size; m++) {
                    for (size_t n = 0; n < size; n++) {
                        res(i / size, j / size, c) = std::max(res(i / size, j / size, c), target(i + m, j + n, c));
                    }
                }
            }
        }
    }
    return res;
}

template<typename T, typename U, typename V>
std::vector<T> fullConnect(const std::vector<T> &flattenedMat, const std::vector<U> &weight, const std::vector<V> &bias) {
#ifndef NDEBUG
    if (flattenedMat.size() != weight.size() / bias.size())
        throw std::invalid_argument(
                "bad connect args: " + std::to_string(flattenedMat.size()) + " " + std::to_string(weight.size()));
#endif
    size_t out = bias.size();
    std::vector<T> res(out);
//#pragma omp parallel for
    for (int o = 0; o < out; o++) {
        for (int i = 0; i < flattenedMat.size(); i++) {
            float w = weight[o * flattenedMat.size() + i];
            res[o] += w * flattenedMat[i];
        }
        res[o] += bias[o];
    }
    return res;
}

template<typename T>
std::vector<T> softMax(std::vector<T> vec) {
    std::vector<T> res(vec.size());
    T sum = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = std::exp(vec[i]);
        sum += vec[i];
    }
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = vec[i] / sum;
    }
    return res;
}
