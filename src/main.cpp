// -*- coding: utf-8 -*-
/*
 This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Aprendizaje 
    National Autonomous University of Mexico
    main.cpp
    @author leguiart
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include "Perceptron.h"
#include "NeuralNet.h"

// Extraer los datos como una matriz de Armadillo de tipo T, si no hay datos, la matriz estará vacía
template<typename T>
arma::Mat<T> load_mat(std::ifstream &file, const std::string &keyword) {
  std::string line;
  std::stringstream ss;
  bool process_data = false;
 	bool has_data = false;
 	while (std::getline(file, line)) {
 		if (line.find(keyword) != std::string::npos) {
 			process_data = !process_data;
 			if (process_data == false) break;
 			continue;
 		}
 		if (process_data) {
 			ss << line << '\n';
 			has_data = true;
 		}
 	}
 
 	arma::Mat<T> val;
 	if (has_data) {
 		val.load(ss);
 	}
 	return val;
}

void save_mat(std::ofstream &file, std::vector<arma::mat> A)
{
  for(int i=0; i<A.size(); i++)
  {
    std::string str;
    for(int j=0; j<A[i].n_cols; j++)
    {
      str.append(std::to_string(arma::as_scalar(A[i](0, j))));
      if(j<A[i].n_cols -1)
        str.append(",");
    }
    file << str << std::endl;
  }  
}


arma::mat p(double, double, int);
arma::mat g(arma::mat, double, double);
 
int main() {
  NeuralNet nn1, nn2, nn3, nn4;
  std::ofstream file;
  std::vector<arma::mat> M;
  std::vector<Perceptron> perceptrones1, perceptrones2, perceptrones3, perceptrones4;
  arma::mat W1, b1, W2, b2, p1, g1, g2, aux;
  W1 << -0.27 << arma::endr
     << -0.41;
  b1 << -0.48 <<arma::endr
     << -0.13;
  W2 << 0.09 << -0.17 << arma::endr;
  b2 << 0.48 << arma::endr;
  p1 = p(-2.0f, 2.0f, 21);
  g1 = g(p1, M_PI/4.0f, 1.0f);
  g2 = g(p1, M_PI/2.0F, 1.0f);
  //creamos cuatro listas de perceptrones para los 4 ejercicios
  perceptrones1.push_back(Perceptron(1, 2, W1, b1, LOG_SIG));
  perceptrones1.push_back(Perceptron(2, 1, W2, b2, LINEAR));
  nn1 = NeuralNet(perceptrones1, p1, g1, 0.1f, 0.01f, 5000, 21);
  nn1.Train();
  aux = nn1.Evaluate();
  std::cout << aux << std::endl;
  std::cout << g1 << std::endl;
  M.push_back(p1);
  M.push_back(g1);
  M.push_back(aux);
  M.push_back(nn1.get_its()[0]);
  M.push_back(nn1.get_its()[1]);
  aux.reset();

  perceptrones2.push_back(Perceptron(1, 2, LOG_SIG));
  perceptrones2.push_back(Perceptron(2, 1, LINEAR));
  nn2 = NeuralNet(perceptrones2, p1, g2, 0.5f, 0.01f, 20000);
  nn2.Train();
  aux = nn2.Evaluate();
  std::cout << aux << std::endl;
  std::cout << g2 << std::endl;
  M.push_back(g2);
  M.push_back(aux);
  aux.reset();

  perceptrones3.push_back(Perceptron(1, 2, LOG_SIG));
  perceptrones3.push_back(Perceptron(2, 1, LINEAR));
  nn3 = NeuralNet(perceptrones3, p1, g2, 0.1f, 0.02f, 50000);
  nn3.Train();
  aux = nn3.Evaluate();
  std::cout << aux << std::endl;
  std::cout << g2 << std::endl;
  M.push_back(aux);
  aux.reset();

  perceptrones4.push_back(Perceptron(1, 10, LOG_SIG));
  perceptrones4.push_back(Perceptron(10, 1, LINEAR));
  nn4 = NeuralNet(perceptrones4, p1, g2, 0.5f, 0.1f, 80000);
  nn4.Train();
  aux = nn4.Evaluate();
  std::cout << aux << std::endl;
  std::cout << g2 << std::endl;
  M.push_back(aux);

  file.open("matlab_data.txt");
  save_mat(file, M);
  file.close();
 	return 0;
}

//Lista de ejemplos
arma::mat p(double inf, double sup, int muest){
  arma::mat Ret(1, muest);
  double delta = (sup-inf)/(muest-1);
  arma::mat aux(1,muest);
  for(int i = 0; i<muest; i++){
    aux(0,i) = inf + delta*i;
  }
  return aux;
}

arma::mat g(arma::mat p, double o, double b){
  arma::mat Ret(1, p.n_cols);
  for(int i = 0; i < p.n_cols; i++){
    Ret(i) = sin(p(0,i)*o) + b;
  }
  return Ret;
}
