#include <algorithm>
#include <vector>

#include "caffe/layers/cosh_quantization_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long in 8/9/2016
namespace caffe{

template <typename Dtype>
void CoshQuantizationLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	dim_ = this->layer_param_.cosh_quantization_loss_param().dim();
	sigmoid_flag_ = this->layer_param_.cosh_quantization_loss_param().sigmoid_flag();
	batch_ = bottom[0]->num();
	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "CoshQuantizationLossLayer: code length must match.";
}

template <typename Dtype>
void CoshQuantizationLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "CoshQuantizationLossLayer: code length must match.";
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}


template <typename Dtype>
void CoshQuantizationLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	Dtype cosh_quantization_loss(0.0);
	const Dtype* bottom_data = bottom[0]->cpu_data();
	for (int n = 0; n < batch_; ++n){
		for (int c = 0; c < dim_; ++c){
			int offset = n*dim_ + c;
			if (!sigmoid_flag_)
				//For the tanh quantizaton
				cosh_quantization_loss +=
				log(cosh(abs(bottom_data[offset]) - 1));
			else // For the sigmoid quantization
				cosh_quantization_loss +=
				log(cosh(abs(Dtype(2.0)*bottom_data[offset] - Dtype(1.0)) - 1));
		}
	}
	top[0]->mutable_cpu_data()[0] = cosh_quantization_loss / batch_;
}

template <typename Dtype>
void CoshQuantizationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();

	if (propagate_down[0]){
		for (int n = 0; n < batch_; ++n){
			for (int c = 0; c < dim_; ++c){
				int offset = n*dim_ + c;
				if (!sigmoid_flag_)
					// For the tanh quantization
					bottom_diff[offset] = Dtype(2.0)*(tanh(abs(bottom_data[offset]) - 1)*
					(bottom_data[offset]>0 ? Dtype(1.0) : Dtype(-1.0))) / batch_;
				else // For the sigmoid quantization
					bottom_diff[offset] = Dtype(4.0)*(tanh(abs(Dtype(2.0)*bottom_data[offset] - Dtype(1.0)) - 1)*
					(Dtype(2.0)*bottom_data[offset] - Dtype(1.0)>0 ? Dtype(1.0) : Dtype(-1.0))) / batch_;
			}
		}
	}
}


INSTANTIATE_CLASS(CoshQuantizationLossLayer);
REGISTER_LAYER_CLASS(CoshQuantizationLoss);
}// namespace caffe
