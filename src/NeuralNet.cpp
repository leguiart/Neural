// -*- coding: utf-8 -*-
/*
    NeuralNet.cpp
    NeuralNet.h
    @author leguiart
*/

#include "NeuralNet.h"

void NeuralNet::Train()
{
  int j = 0; //Contador de iteraciones
  arma::mat eval;
  double error_mc;
  do{
    error_mc = 0;
    //Para cada capa
    for(int i = 0; i < layerList.size(); i++)
    {
      //Si es la ultima capa
      if(i == layerList.size() - 1)
      {
        arma::mat S(layerList[i].getOutputs(),1); //Reservando espacio de Sensibilidad de la ultima capa
        s.push_back(S); //Agrega a la lista de sensibilidades
        continue;
      }
      arma::mat S(layerList[i].getOutputs(), layerList[i].getOutputs()); //sensibilidad de la capa iesima
      s.push_back(S);
    }
    if(cont_init>-1 && j==0)
      cont = cont_init-1;
    else
      cont = rand()%(A.size()-1);

    ForwardProp();
    BackProp();
    if(j<2)
    {
      its.push_back(Evaluate());
      eval = its[j];
    }
    else{
      eval = Evaluate();
    }

    for(int k = 0; k < A.n_cols; k++)
    {
      error_mc += pow(eval[k] - B[k],2);
    }
    error_mc = error_mc/A.n_cols;
    s.clear();
    am.clear();
    j++;
  }while(j<it && error_mc>error_min);
}

void NeuralNet::ForwardProp()
{
  am.push_back(A.col(cont));
  am.push_back(layerList[0].ForwardProp(A, cont));
  for(int i=1; i<layerList.size(); i++)
  {
    am.push_back(layerList[i].ForwardProp(am[i], 0));
  }
  error = B.col(cont) - am[layerList.size()];
}

void NeuralNet::BackProp()
{
  arma::mat W_old, b_old;
  for(int i = layerList.size(); i > 0; i--)
  {
    if(i == layerList.size())
    {
      s[i-1] = -2.0f*layerList[i-1].derivative(am[i])*error;
    }
    else
    {
      s[i-1] = arma::diagmat(layerList[i-1].derivative(am[i]))*(layerList[i].getWeight().t())*s[i];
    }
  }
  for(int i = 0; i <layerList.size(); i++)
  {
    W_old = layerList[i].getWeight();
    b_old = layerList[i].getBias();
    layerList[i].setWeight(W_old - lr*s[i]*am[i].t());
    layerList[i].setBias(b_old - lr*s[i]);
  }
}

arma::mat NeuralNet::Evaluate(){
  arma::mat R;
  R.set_size(size(B));
  for(int i=0; i<B.n_cols; i++)
  {
    am.push_back(A.col(i));
    am.push_back(layerList[0].ForwardProp(A, i));
    for(int j=1; j<layerList.size(); j++)
    {
      am.push_back(layerList[j].ForwardProp(am[j], 0));
    }
    R(0,i) = am[layerList.size()](0,0);
    am.clear();
  }
  return R;
}
