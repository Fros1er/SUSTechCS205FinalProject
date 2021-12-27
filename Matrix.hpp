#pragma once

#include "MatrixImpl.hpp"
#include <memory>
#include <iostream>
#include <sstream>
#include <cmath>

template<typename T>
class Matrix {
private:
    std::vector<std::shared_ptr<MatrixImpl<T>>> pImpls;
    size_t rows, cols, channels, offsetX = 0, offsetY = 0;

    template<typename U>
    bool isValueEqual(const Matrix<U> &rhs) const {
        if (channels != rhs.channels) return false;
        if (rows != rhs.rows || cols != rhs.cols) return false;
        for (size_t i = 0; i < channels; i++) {
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                if (!isEqual(pImpls[i]->base[j], rhs.pImpls[i]->base[j])) return false;
            }
        }
        return true;
    }

    template<typename U>
    void copyImpl(const Matrix<U> &rhs) {
        for (size_t i = 0; i < channels; i++) {
            pImpls[i] = std::make_shared<MatrixImpl<T>>(*rhs.pImpls[i],
                                                        Range{offsetY, offsetY + rows}, Range{offsetX, offsetX + cols});
        }
    }

    // static methods for operator >>
    static void pushToImplBase(std::shared_ptr<MatrixImpl<T>> &impl, T &val) {
        impl->base.push_back(val);
    }

    static void setImplSize(std::shared_ptr<MatrixImpl<T>> &impl, size_t rows, size_t cols) {
        impl->rows = rows;
        impl->cols = cols;
    }

public:

    Matrix(const Matrix &rhs) : pImpls(rhs.channels), rows{rhs.rows},
                                cols{rhs.cols}, channels{rhs.channels}, offsetX{rhs.offsetX}, offsetY{rhs.offsetY} {
        copyImpl(rhs);
    }

    template<typename U>
    explicit Matrix(const Matrix<U> &rhs) : pImpls(rhs.channels), rows{rhs.rows},
                                            cols{rhs.cols}, channels{rhs.channels}, offsetX{rhs.offsetX},
                                            offsetY{rhs.offsetY} {
        copyImpl(rhs);
    }

    Matrix(Matrix<T> &&rhs) noexcept = default;

    explicit Matrix(const size_t &rows = 0, const size_t &cols = 0, const size_t channels = 1, const T &val = T())
            : pImpls(channels), rows{rows}, cols{cols}, channels{channels} {
        for (auto &v: pImpls) {
            v = std::make_shared<MatrixImpl<T>>(rows, cols, val);
        }
    }

    Matrix(const Matrix &other, const Range &row, const Range &col)
            : pImpls(other.pImpls), rows(row.size()), cols(col.size()), channels{other.channels}, offsetX(row.start),
              offsetY(col.start) {
#ifndef NDEBUG
        if (rows + offsetX > pImpls[0]->getNumberOfRows() || cols + offsetY > pImpls[0]->getNumberOfCols())
            throw std::invalid_argument("Invalid Range");
#endif
    }

    std::vector<T> flat() const {
        std::vector<T> res;
        for (auto &v: pImpls) {
            res.insert(res.end(), v->base.begin(), v->base.end());
        }
        return res;
    }

    Matrix<T> &operator=(const Matrix<T> &rhs) {
        Matrix<T> temp(rhs);
        swap(temp);
        return *this;
    }

    Matrix<T> &operator=(Matrix<T> &&rhs) noexcept = default;

    std::vector<std::reference_wrapper<T>> operator()(const size_t &row, const size_t &col) {
#ifndef NDEBUG
        if (row >= rows || col >= cols) throw std::invalid_argument("invalid row or col");
#endif
        std::vector<std::reference_wrapper<T>> res;
        for (std::shared_ptr<MatrixImpl<T>> &v: pImpls) {
            res.push_back(v->operator()(offsetX + row, offsetY + col));
        }
        return res;
    }

    std::vector<T> operator()(const size_t &row, const size_t &col) const {
#ifndef NDEBUG
        if (row >= rows || col >= cols) throw std::invalid_argument("invalid row or col");
#endif
        std::vector<T> res(channels);
        for (size_t i = 0; i < channels; i++) {
            res[i] = pImpls[i]->operator()(offsetX + row, offsetY + col);
        }
        return res;
    }

    T &operator()(const size_t row, const size_t col, const size_t channel) {
#ifndef NDEBUG
        if (row >= rows || col >= cols) throw std::invalid_argument("invalid row or col");
#endif
        return (*pImpls[channel])(row + offsetX, col + offsetY);
    }

    const T &operator()(const size_t row, const size_t col, const size_t channel) const {
#ifndef NDEBUG
        if (row >= rows || col >= cols) throw std::invalid_argument("invalid row or col");
#endif
        return (*pImpls[channel])(row + offsetX, col + offsetY);
    }

    Matrix operator()(const Range &row, const Range &col) const {
        return Matrix(*this, row, col);
    }

    bool operator==(const Matrix &rhs) const {
        if (this == &rhs) return true;
        if (channels != rhs.channels) return false;
        bool pImplFlag = true;
        for (size_t i = 0; i < channels; i++) {
            if (pImpls[i] != rhs.pImpls[i]) {
                pImplFlag = false;
                break;
            }
        }
        if (pImplFlag && rows == rhs.rows && cols == rhs.cols && offsetX == rhs.offsetX && offsetY == rhs.offsetY)
            return true;
        return isValueEqual(rhs);
    }

    template<typename U>
    bool operator==(const Matrix<U> &rhs) const {
        return isValueEqual(rhs);
    }

    template<typename U>
    auto operator+(const Matrix<U> &rhs) const {
#ifndef NDEBUG
        if (rows != rhs.rows || cols != rhs.cols) throw std::invalid_argument("A.size != B.size");
        if (channels != rhs.channels)
            throw std::invalid_argument("A.channel != B.channel");
#endif
        typedef decltype(T() + U()) return_type;
        Matrix<return_type> res(rows, cols, channels);
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                res.pImpls[i]->base[j] = pImpls[i]->base[j] + rhs.pImpls[i]->base[j];
            }
        }

        return res;
    }

    template<typename U>
    Matrix<T> &operator+=(const Matrix<U> &rhs) {
#ifndef NDEBUG
        if (rows != rhs.rows || cols != rhs.cols) throw std::invalid_argument("A.size != B.size");
        if (channels != rhs.channels)
            throw std::invalid_argument("A.channel != B.channel");
#endif
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                pImpls[i]->base[j] += rhs.pImpls[i]->base[j];
            }
        }
        return *this;
    }

    Matrix<T> &operator+() {
        return *this;
    }

    template<typename U>
    auto operator-(const Matrix<U> &rhs) const {
#ifndef NDEBUG
        if (rows != rhs.rows || cols != rhs.cols) throw std::invalid_argument("A.size != B.size");
        if (channels != rhs.channels)
            throw std::invalid_argument("A.channel != B.channel");
#endif
        typedef decltype(T() - U()) return_type;
        Matrix<return_type> res(rows, cols, channels);
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                res.pImpls[i]->base[j] = pImpls[i]->base[j] - rhs.pImpls[i]->base[j];
            }
        }
        return res;
    }

    template<typename U>
    Matrix<T> &operator-=(const Matrix<U> &rhs) {
#ifndef NDEBUG
        if (rows != rhs.rows || cols != rhs.cols) throw std::invalid_argument("A.size != B.size");
        if (channels != rhs.channels)
            throw std::invalid_argument("A.channel != B.channel");
#endif
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                pImpls[i]->base[j] -= rhs.pImpls[i]->base[j];
            }
        }
        return *this;
    }

    Matrix<T> operator-() const {
        Matrix<T> res(rows, cols, channels);
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                res.pImpls[i]->base[j] = -pImpls[i]->base[j];
            }
        }
        return res;
    }

    template<typename U>
    Matrix<T> operator*(const U &rhs) const {
        Matrix<T> res(rows, cols, channels);
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                res.pImpls[i]->base[j] = pImpls[i]->base[j] * rhs;
            }
        }
        return res;
    }

    template<typename U>
    auto operator*(const Matrix<U> &rhs) const {
#ifndef NDEBUG
        if (rows != rhs.cols) throw std::invalid_argument("A.rows != B.cols");
        if (channels != rhs.channels)
            throw std::invalid_argument("A.channel != B.channel");
#endif
        typedef decltype(T() + U()) return_type;
        Matrix<return_type> res(rows, rhs.cols, channels);
        for (size_t c = 0; c < channels; c++) {
#pragma omp parallel for
            for (size_t i = 0; i < rows; i++) {
                for (size_t k = 0; k < cols; k++) {
                    const T &tmp = (*this)(i, k, c);
                    for (size_t j = 0; j < rhs.cols; j++) {
                        res(i, j, c) += tmp * rhs(k, j, c);
                    }
                }
            }
        }
        return res;
    }

    template<typename U>
    Matrix<T> &operator*=(const U &rhs) {
        for (size_t i = 0; i < channels; i++) {
#pragma omp parallel for
            for (size_t j = 0; j < pImpls[i]->base.size(); j++) {
                pImpls[i]->base[j] *= rhs;
            }
        }
        return *this;
    }

    void swap(Matrix<T> &other) {
        using std::swap;
        pImpls.swap(other.pImpls);
        swap(rows, other.rows);
        swap(cols, other.cols);
        swap(offsetY, other.offsetY);
        swap(offsetX, other.offsetX);
        swap(channels, other.channels);
    }

    friend std::istream &operator>>(std::istream &in, Matrix &mat) {
        for (size_t i = 0; i < mat.channels; i++) {
            mat.pImpls[i] = std::make_shared<MatrixImpl<T>>();
        }
#ifndef NDEBUG
        bool first = true;
#endif
        size_t rowCount = 0;
        for (std::string line; std::getline(in, line);) {
            std::istringstream iss(line);
            size_t channel = 0;
            size_t colCount = 0;
            rowCount++;
            for (T num; iss >> num;) {
                pushToImplBase(mat.pImpls[channel++], num);
                channel %= mat.channels;
                colCount++;
            }
#ifndef NDEBUG
            if (channel != 0) {
                throw std::invalid_argument("wrong channel numbers.");
            }
            if (!first && mat.cols != colCount / mat.channels) {
                throw std::invalid_argument("wrong input.");
            }
            first = false;
#endif
            for (size_t i = 0; i < mat.channels; i++) {
                setImplSize(mat.pImpls[i], rowCount, colCount / mat.channels);
            }
            mat.cols = colCount / mat.channels;
        }
        mat.rows = rowCount;
        mat.offsetY = 0;
        mat.offsetX = 0;
        return in;
    }

    size_t getNumberOfRows() const {
        return rows;
    }

    size_t getNumberOfCols() const {
        return cols;
    }

    size_t getNumberOfChannels() const {
        return channels;
    }

    friend std::ostream &operator<<(std::ostream &out, const Matrix<T> &mat) {
        for (size_t i = 0; i < mat.rows; i++) {
            for (size_t j = 0; j < mat.cols; j++) {
                for (size_t k = 0; k < mat.channels; k++) {
                    out << mat(i, j, k) << " ";
                }
            }
            out << std::endl;
        }
        return out;
    }

    template<typename> friend
    class Matrix;

    template<typename> friend
    class MatrixImpl;
};

template<typename T>
void swap(Matrix<T> &a, Matrix<T> &b) {
    a.swap(b);
}
