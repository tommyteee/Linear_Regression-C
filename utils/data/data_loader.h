#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;

class DataFrame{
    public:
        Eigen::MatrixXd data;

    public:
        DataFrame();
        DataFrame(Eigen::MatrixXd data) : data(data){}
        ~DataFrame();

        void display(){
            std::cout << this->data << std::endl; 
        };
        static DataFrame read_csv(const string& filename, const char& delimiter);

        static Eigen::MatrixXd from_vector(std::vector<std::vector<double>> vect);

        std::tuple<DataFrame, DataFrame> split(const int y_index);

        static DataFrame validate_data(DataFrame X);

        std::tuple<int , int> shape(){
            return {this->data.rows(), this->data.cols()};
        };
};

#endif