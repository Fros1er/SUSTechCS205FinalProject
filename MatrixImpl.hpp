#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>

template<typename L, typename R>
std::enable_if_t<std::disjunction_v<std::is_floating_point<L>, std::is_floating_point<R>>, bool>
isEqual(const L &lhs, const R &rhs) {
    return std::abs(lhs - rhs) <=
           (((std::abs(rhs) < std::abs(lhs)) ? std::abs(rhs) : std::abs(lhs)) * std::numeric_limits<double>::epsilon());
}

template<typename L, typename R>
std::enable_if_t<!std::disjunction_v<std::is_floating_point<L>, std::is_floating_point<R>>, bool>
isEqual(const L &lhs, const R &rhs) {
    return lhs == rhs;
}

struct Range {
    size_t start, end;

    Range(size_t start, size_t end) : start(start), end(end) {
#ifndef NDEBUG
        if (start > end) throw std::invalid_argument("start > end");
#endif
    }

    [[nodiscard]] size_t size() const {
        return end - start;
    }
};

template<typename T>
class MatrixImpl {
private:
    size_t rows, cols;
    std::vector<T> base;

public:

    template<typename U>
    explicit MatrixImpl(const MatrixImpl<U> &rhs, const Range row, const Range col)
            : rows{row.size()}, cols{col.size()}, base(row.size() * col.size()) {
        for (size_t i = 0; i < base.size(); i++) {
            base[i] = static_cast<T>(rhs.base[i]);
        }
    }

    explicit MatrixImpl(const size_t &row = 0, const size_t &col = 0, const T &val = T())
            : rows{row}, cols{col}, base(row * col, val) {}

    [[nodiscard]] size_t getNumberOfRows() const {
        return rows;
    }

    [[nodiscard]] size_t getNumberOfCols() const {
        return cols;
    }

    T &operator()(const size_t row, const size_t col) {
        return base[row * cols + col];
    }

    T operator()(const size_t row, const size_t col) const {
        return base[row * cols + col];
    }

    template<typename> friend
    class MatrixImpl;

    template<typename> friend
    class Matrix;
};
