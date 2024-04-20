#include "multiple.h"
#include "../data/data_loader.h"
#include "linear_regression.h"
#include <omp.h>

LinearRegressionMultiple::~LinearRegressionMultiple(){}


void LinearRegressionMultiple::fit(DataFrame X_train, DataFrame y_train){

    auto [n_samples , n_features] = X_train.shape();

    auto [_ , y_dim] = y_train.shape();

    weights = Eigen::MatrixXd::Random(n_features + 1, y_dim);
    coef_ = Eigen::MatrixXd::Random(n_features, y_dim);
    intercept_ = Eigen::MatrixXd::Random(1 , y_dim);

    #pragma omp parallel for
    for(int j = 0; j < y_dim ; j++){
        
        LinearRegression lg(lr, n_iters);

        Eigen::Matrix<double, Eigen::Dynamic, 1> y_trainVect = y_train.data.col(j);

        lg.fit(X_train, DataFrame(y_trainVect));

        #pragma omp critical
        {
            intercept_(0, j) = lg.intercept_(0, 0);
            coef_.col(j) = lg.coef_.col(0);
        }
    }
}