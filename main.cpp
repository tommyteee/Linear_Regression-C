#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils/data/data_loader.h"
#include "utils/linear_models/linear_regression.h"
#include "utils/linear_models/multiple.h"

using namespace Eigen;

int main(int argc, char* argv[]){

    DataFrame df = DataFrame::read_csv("./data/mulg_data(1).csv", ',');

    auto [X_train, y_train] = df.split(1);

    LinearRegressionMultiple lgm(0.01, 1000);

    lgm.fit(X_train, y_train);

    std::cout << "coef_ : \n"<< lgm.coef_ << std::endl;
    std::cout << "intercept_ :\n" << lgm.intercept_ << std::endl;

}