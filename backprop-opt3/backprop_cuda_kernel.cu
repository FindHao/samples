

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid) 
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
  //  __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];

   weight_matrix[ty][tx] = input_hidden_cuda[index] * input_cuda[index_in];

   __syncthreads();   

  if((ty & 1) ==0){
    weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 1][tx];
    if((ty & 3) ==0){
      weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 2][tx];
      if((ty & 7) ==0){
        weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 4][tx];
        if((ty & 15) ==0){
          weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 8][tx];
        }
      }
    }
  }
    


   input_hidden_cuda[index] = weight_matrix[ty][tx];

   if (tx == 0) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}



__global__ void bpnn_adjust_weights_cuda(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif 
