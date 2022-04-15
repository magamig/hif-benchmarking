/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : 
 * Module      : 
 * Class       : none
 * Language    : C++
 * Description : 
 *
 * Author     : Thomas Pock
 * EMail      : pock@icg.tugraz.at
 *
 *
 * This software is based on work of Yunmei Chen and Xiaojing Ye.
 * Projection Onto A Simplex
 * http://arxiv.org/abs/1101.6081
 *
 */
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <mex.h>

// #include <omp.h>

///////////////////////////////////////////////////////////
// compile with: mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" reproject_simplex_mex_fast.cpp
///////////////////////////////////////////////////////////


typedef double ElementType;


void Swap( ElementType *Lhs, ElementType *Rhs )
{
  ElementType Tmp = *Lhs;
  *Lhs = *Rhs;
  *Rhs = Tmp;
}
ElementType Median3( ElementType A[ ], int Left, int Right )
{
  int Center = ( Left + Right ) / 2;
  
  if( A[ Left ] > A[ Center ] )
    Swap( &A[ Left ], &A[ Center ] );
  if( A[ Left ] > A[ Right ] )
    Swap( &A[ Left ], &A[ Right ] );
  if( A[ Center ] > A[ Right ] )
    Swap( &A[ Center ], &A[ Right ] );
  
  /* Invariant: A[ Left ] <= A[ Center ] <= A[ Right ] */
  
  Swap( &A[ Center ], &A[ Right - 1 ] );  /* Hide pivot */
  return A[ Right - 1 ];                /* Return pivot */
}
void InsertionSort( ElementType A[ ], int N )
{
  int j, P;
  ElementType Tmp;
  
  /* 1*/      for( P = 1; P < N; P++ ) {
    /* 2*/          Tmp = A[ P ];
    /* 3*/          for( j = P; j > 0 && A[ j - 1 ] > Tmp; j-- )
      /* 4*/              A[ j ] = A[ j - 1 ];
    /* 5*/          A[ j ] = Tmp;
  }
}

#define Cutoff ( 3 )

void Qsort( ElementType A[ ], int Left, int Right )
{
  int i, j;
  ElementType Pivot;
  if( Left + Cutoff <= Right )
  {
    Pivot = Median3( A, Left, Right );
    i = Left; j = Right - 1;
    for( ; ; )
    {
      while( A[ ++i ] < Pivot ){ }
      while( A[ --j ] > Pivot ){ }
      if( i < j )
        Swap( &A[ i ], &A[ j ] );
      else
        break;
    }
    Swap( &A[ i ], &A[ Right - 1 ] );  /* Restore pivot */
    
    Qsort( A, Left, i - 1 );
    Qsort( A, i + 1, Right );
  }
  else  /* Do an insertion sort on the subarray */
    InsertionSort( A + Left, Right - Left + 1 );
}


void Quicksort( ElementType A[ ], int N )
{
  Qsort( A, 0, N - 1 );
}



void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) 
{
  // omp_set_num_threads(16);

  double* u =  (double*)mxGetPr(prhs[0]);
  const mwSize* dims = mxGetDimensions(prhs[0]);
  int height = dims[0];
  int width = dims[1];
  int labels = dims[2];
 
  // std::cout << "height = " << height << ", width = " << width
  //           << ", labels = " << labels << ", components = " << dims[3] << std::endl;
  
#pragma omp parallel for
  for (int x=0; x<width*height; x++) {
    double v[40];
    double s[40];


    // load the data
    for (int z=0; z<labels; z++) {
      v[z] = u[x + z*width*height];
      s[z] = v[z];
    }
    
    double sumResult = -1, tmpValue, tmax; 
    bool bget = false;
    
    Quicksort(s,labels);
    for(int j = labels-1; j >= 1; j--){    	
      sumResult = sumResult + s[j];
      tmax = sumResult/(labels-j);
      if(tmax >= s[j-1]){
        bget = true;
        break;
      }
    }
    
    /* if t is less than s[0] */
    if(!bget){
      sumResult = sumResult + s[0];
      tmax = sumResult/labels;
    }
    
    // store the data
    for (int z=0; z<labels; z++) {
      u[x + z*width*height] = std::max((double)0, v[z]-tmax);
    }
    
  }

    // assign input to output
  //plhs[0] = (mxArray*)prhs[0];
  plhs[0] = mxDuplicateArray(prhs[0]);

}
