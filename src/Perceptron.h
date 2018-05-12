#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <iostream>
#include <vector>
/****Definicion de estructura tipo****/
/*!
  Esta estructura contendra cada uno de
  los ejemplos, con su target correspondiente
*/
typedef struct Percept{
  arma::mat p; /*! ejemplo */
  arma::mat t; /*! target */
}Percept;

enum TransferFunction{
  HARD_LIM,
  LOG_SIG,
  LINEAR
};

class Perceptron {
  public:
    /****! Constructor1 ****/
    /*!
      Constructor vacio para
      instanciar la clase sin parametros
    */
    Perceptron(){
    }

    /****! Constructor2 ****/
    /*!
      Constructor con parametros para
      inicializar el objeto
      @param A: matriz de categorías, ejemplos y targets
      @param B: vector renglón de número de ejemplos
      @param W: Matriz de pesos inicial
      @param b: Bias inicial
    */
    Perceptron(arma::mat, arma::mat, arma::mat, arma::mat, TransferFunction tf);

    /****! Constructor3 ****/
    /*!
      Constructor para backpropagation
      @params n, m: numero de entradas, numero de salidas
      @param W: Matriz de pesos inicial
      @param b: Bias inicial
      @param tf: funcion de transferencia asociada a la capa
    */
    Perceptron(int n, int m, arma::mat W, arma::mat b, TransferFunction tf) :
    n(n), m(m), W(W), b(b), tf(tf)
    {
    }

    /****! Constructor4 ****/
    /*!
      Constructor para backpropagation que señaliza inicializar pesos y bias aleatoriamente
      @params int, int: numero de entradas, numero de salidas
      @param TransferFunction: funcion de transferencia asociada a la capa
    */
    Perceptron(int, int, TransferFunction);

    /****! Metodo publico ****/
    /*!
      Este metodo evalua la entrada a la capa saca 'n', como salida f(n)
      @params mat: matriz de entradas a la capa
      @param int: punto aleatorio a evaluar 
    */
    arma::mat ForwardProp(arma::mat, int);

    /****! Metodo publico ****/
    /*!
      regresa la matriz evaluada en la derivada de la funcion de transferencia
      para backpropagation
      @params mat: matriz de salidas de las capaz
    */
    arma::mat derivative(arma::mat);

    /****! Metodo publico ***/
    /*!
      Establece la matriz de pesos
      @params W
    */
    void setWeight(arma::mat W)
    {
      this->W = W;
    }

    /****! Metodo publico ***/
    /*!
      Establece la matriz de bias
      @params
    */
    void setBias(arma::mat b)
    {
      this->b = b;
    }
    
    /****! Método miembro público ****/
    /*!
      Método donde se aplica la regla
      de aprendizaje del perceptrón y se obtienen
      los pesos y bias finales
    */
    void Train();

    /****! Método miembro público ****/
    /*!
      Método getter de los pesos
      @return W: Matriz de pesos
    */
    arma::mat getWeight();

    /****! Método miembro público ****/
    /*!
      Método getter del bias
      @return b: Bias
    */
    arma::mat getBias();

    /****! Metodo publico ***/
    /*!
      Método getter del numero de entradas
      @return n: entradas
    */
    int getInputs()
    {
      return n;
    }

    /****! Metodo publico ***/
    /*!
      Método getter del numero de salidas
      @return m: salidas
    */
    int getOutputs()
    {
      return m;
    }

    /****! Metodo publico ***/
    /*!
      Método getter de la funcion de transferencia de la capa
      @params
    */
    TransferFunction getTf()
    {
      return tf;
    }

  protected:
    /****! Atributos protegidos enteros ****/
    /*!
      n: Número de entradas
      m: Número de salidas
      c: Número de categorías
      e: Número de ejemplos por categoría
    */  
    int n, m, c, e;

    /****! Atributos protegidos tipo matriz de armadillo ****/
    /*!
      W: Matriz de pesos
      A: Matriz de categorías, ejemplos y targets
      B: Vector renglón con [n,m,c,e]
      b: Vector de bias
    */
    arma::mat W, A, B, b;

    /****! Atributo protegido tipo vector de Percept ****/
    /*!
      En este vector de tipos Percept se guardará cada uno de los
      ejemplos con su target correspondiente
    */
    std::vector<Percept> percept;

    //Funcion de transferencia, objeto enum 
    TransferFunction tf;

    /****! Método miembro protegido ****/
    /*!
      Método que calculará el hardlim de una matriz
      @return a: Resultado de la función de transferencia
    */
    arma::mat hardLim(arma::mat);

    /****! Método miembro protegido ****/
    /*!
      Método que calculará el logsig de una matriz
      @return a: Resultado de la función de transferencia
    */
    arma::mat logSig(arma::mat);

    /****! Método miembro protegido ****/
    /*!
      Método que calculará la Wp + b
      @return n: Resultado de Wp + b
    */
    arma::mat weightedSum(arma::mat, arma::mat, arma::mat);
};

#endif
