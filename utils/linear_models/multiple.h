#ifndef MULTIPLE_H
#define MULTIPLE_H
#include "../data/data_loader.h"

class LinearRegressionMultiple{

    public:
        double lr;
        int n_iters;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coef_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> intercept_;

    private:
        Eigen::MatrixXd weights;

    public:
        LinearRegressionMultiple(double lr , int n_iters): lr(lr), n_iters(n_iters){};
        ~LinearRegressionMultiple();

        void fit(DataFrame X_train, DataFrame y_train);
};


#endif 