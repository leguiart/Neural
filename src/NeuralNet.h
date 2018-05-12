#ifndef NEURALNET_H_
#define NEURALNET_H_

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <iostream>
#include <time.h>
#include <vector>

#include "Perceptron.h"

class NeuralNet{
    public:
    /****! Constructor1 ****/
    /*!
        Constructor vacio para
        instanciar la clase sin parametros
    */
        NeuralNet(){
        }

    /****! Constructor2 ****/
    /*!
        Constructor con parametros para
        inicializar el objeto
        @param layerList: vector de perceptrones 
        @param A: Matriz de puntos 
        @param B: Matriz de funcion evaluada en los puntos
        @param lr: Learning Rate
        @param it: numero de iteraciones a efectuar 
    */
        NeuralNet(std::vector<Perceptron> layerList, arma::mat A, arma::mat B, double lr, double error_min, int it) :
        layerList(layerList), A(A), B(B),lr(lr), error_min(error_min), it(it){
            srand(time(NULL));
            cont_init = -1;
        }

    /****! Constructor3 ****/
    /*!
        Constructor con parametros para
        inicializar el objeto
        @param layerList: vector de perceptrones 
        @param A: Matriz de puntos 
        @param B: Matriz de funcion evaluada en los puntos
        @param lr: Learning Rate
        @param it: numero de iteraciones a efectuar 
    */
        NeuralNet(std::vector<Perceptron> layerList, arma::mat A, arma::mat B, double lr, double error_min, int it, double cont_init) :
        layerList(layerList), A(A), B(B),lr(lr), error_min(error_min), it(it), cont_init(cont_init){
            srand(time(NULL));
        }

    /****! Metodo publico ****/
    /*!
        Método que entrena a la red y aplica el algoritmo de forward propagation y backpropagation
        tarea que se delega a otras funciones
    */
        void Train();
    
     /****! Metodo publico ****/
    /*!
        Método que Evalua a la red despues de un entrenamiento
        @return mat: Regresa todas las salidas de la red dados todos los puntos de entrada
    */
        arma::mat Evaluate();

    /****! Metodo publico ****/
    /*!
        @return vector<mat>: Regresa la red evaluada en cierta cantidad de iteraciones
    */
        std::vector<arma::mat> get_its(){
            return its;
        }
    protected:
        /****! Atributo protegido vector<Perceptron>****/
        /*!
            vector<Perceptron>: lista de perceptrones, esta lista tendrá 
            la informacion de cada una de las capas que componen a la red
        */  
        std::vector<Perceptron> layerList;

        /****! Atributos protegidos vector<mat> ****/
        /*!
            vector<mat> am: contendra las salidas de cada capa y la entrada y salida de la red
            vector<mat> s: contendra las matrices de sensibilidad de cada capa
            vector<mat> its: contendra las salidas de cada entrada posible para cualquier numero de iteraciones en el 
                entrenamiento
        */  
        std::vector<arma::mat> am, s, its;

        /****! Atributos protegidos mat ****/
        /*!
            mat A: contendra los puntos a evaluar
            mat B: contendra la evaluacion de los puntos en la funcion original
            mat error: contendra el error de la red, la referencia menos la salida de la red
            mat R: contendra la salida de la red con los pesos y bias de la iteracion final
        */  
        arma::mat A, B, error;

        /****! Atributos protegidos enteros ****/
        /*!
            it: Número total de iteraciones a realizar
            cont_init: punto de la primera iteración 
            cont: iterador de los puntos de entrada
        */  
        int it, cont_init, cont;

         /****! Atributos protegidoo double ****/
        /*!
            lr: learning rate
        */  
        double lr, error_min;

        /****! Metodo protegido ****/
        /*!
            Procedimiento que sera llamado para efectuar el paso de forward propagation
        */
        void ForwardProp();

        /****! Metodo protegido ****/
        /*!
            Procedimiento que sera llamado para efectuar el paso de backpropagation,
            y actualizara los pesos y bias
        */
        void BackProp();
    private:
        
     
};

#endif
