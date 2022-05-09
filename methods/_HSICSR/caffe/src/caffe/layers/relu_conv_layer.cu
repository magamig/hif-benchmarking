#include <vector>

#include "caffe/layers/relu_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReluConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  Dtype* weight_ = this->blobs_[0]->mutable_cpu_data();
  const int count = this->blobs_[0]->count();
  const int num = this->blobs_[0]->num();
  const int channels = this->blobs_[0]->channels();
  
  Dtype sum = 0;
  for (int i = 0; i < num; ++i) {
    sum = 0;
    for (int j = 0; j < channels; ++j) {
      if (weight_[i*channels+j] < 0) {
        weight_[i*channels+j] = 0;
      } else {
        sum += weight_[i*channels+j];
      }
    }
    for (int j = 0; j < channels; ++j) {
      weight_[i*channels+j] /= sum;
      cudaMemcpy(weight+i*channels+j, weight_+i*channels+j, sizeof(Dtype), cudaMemcpyHostToDevice);
    }
  }
  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, (const Dtype*)weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ReluConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReluConvolutionLayer);
//REGISTER_LAYER_CLASS(ReluConvolution);

}  // namespace caffe
