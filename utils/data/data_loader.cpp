#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include "data_loader.h"

using namespace std;


// DataFrame::DataFrame(Eigen::Matrix2Xd data){

// }

DataFrame::~DataFrame(){}

DataFrame::DataFrame(){}

Eigen::MatrixXd DataFrame::from_vector(std::vector<std::vector<double>> vect){

    Eigen::MatrixXd matrix(vect.size(), vect[0].size());
    for (int i = 0; i < vect.size(); ++i) {
        for (int j = 0; j < vect[0].size(); ++j) {
            matrix(i, j) = vect[i][j];
        }
    }

    return matrix;

}


DataFrame DataFrame::read_csv(const string& filename, const char& delimiter){

    std::ifstream file(filename);
    std::vector<std::vector<double>> values;
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Error occured while opening file: " + filename);
    }


    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::vector<double> row;
        std::string value;
        while (std::getline(lineStream, value, delimiter)) {
            row.push_back(std::stod(value));
        }
        values.push_back(row);
    }

    Eigen::MatrixXd data = DataFrame::from_vector(values);

    return DataFrame(data);

}

std::tuple<DataFrame, DataFrame> DataFrame::split(const int y_index){

    Eigen::MatrixXd X_train(data.rows(), data.cols() - y_index -1);
    Eigen::MatrixXd y_train(data.rows(), y_index + 1);

    for(int i = 0; i < data.rows(); i++){
        for(int j = 0; j < data.cols(); j++){
            if(j < data.cols() - y_index - 1){
                X_train(i, j) = data(i, j);
            }else 
                y_train(i, j + y_index - data.cols() +1) = data(i, j);
        }   
    }

    return {DataFrame(X_train),DataFrame(y_train)};
}

DataFrame DataFrame::validate_data(DataFrame X){

    auto [rows, cols] = X.shape();

    Eigen::MatrixXd augmentedMatrix(rows, cols + 1);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(rows);
    augmentedMatrix << ones, X.data;

    return DataFrame(augmentedMatrix);

}