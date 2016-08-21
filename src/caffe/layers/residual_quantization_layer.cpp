#include <cmath>
#include <vector>

#include "caffe/layers/residual_quantization_layer.hpp"


namespace caffe{

/** 
 * Residual quantization loss for hashing learning
 * Added by Fuchen Long in 8/7/2016
*/
template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	sigmoid_flag_ = this->layer_param_.residual_quantization_param().sigmoid_flag();
	lamda_ = this->layer_param_.residual_quantization_param().lamda();
	batch_ = bottom[0]->num();
	dim_ = bottom[0]->channels();
	CHECK_EQ(bottom[1]->num(), batch_)
		<< "Reidual Qunatization Layer: loss and hash code should have the same batch size.";
	CHECK_EQ(bottom[1]->channels(),dim_)
		<< "Reidual Qunatization Layer: loss and hash code should have the same dimension.";
}

template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::quantization_add_forward_cpu_sig(
	const Dtype* code, const Dtype* loss,
	Dtype* res_code){
	for (int pos = 0; pos < batch_*dim_; ++pos){
		if (code[pos] > 0.5)
			res_code[pos] = std::min(Dtype(1.0), code[pos] + loss[pos] / 2);
		else res_code[pos] = std::max(Dtype(0.0), code[pos] - loss[pos] / 2);
	}
}

template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::quantization_add_forward_cpu_tanh(
	const Dtype* code, const Dtype* loss, Dtype* res_code){
	for (int pos = 0; pos < batch_*dim_; ++pos){
		if (code[pos] > 0)
			res_code[pos] = std::min(Dtype(1.0), code[pos] + loss[pos] / lamda_);
		else res_code[pos] = std::max(Dtype(-1.0), code[pos] - loss[pos] / lamda_);
	}
}


template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::quantization_add_backward_cpu_sig(
	const Dtype* code, const Dtype* loss,
	const Dtype* top_diff,const Dtype* res_code,
	Dtype* code_diff, Dtype* loss_diff){
	for (int pos = 0; pos < batch_*dim_; ++pos){
		if (code[pos] > 0.5){
			if (res_code[pos] < 1){
				code_diff[pos] = top_diff[pos];
				loss_diff[pos] = top_diff[pos] / 2;
			}
			else{
				code_diff[pos] = Dtype(0.0);
				loss_diff[pos] = Dtype(0.0);
			}
		}
		else{
			if (res_code[pos] > 0){
				code_diff[pos] = top_diff[pos];
				loss_diff[pos] = -top_diff[pos] / 2;
			}
			else{
				code_diff[pos] = Dtype(0.0);
				loss_diff[pos] = Dtype(0.0);
			}
		}
	}
}

template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::quantization_add_backward_cpu_tanh(
	const Dtype* code, const Dtype*  loss,
	const Dtype* top_diff, const Dtype* res_code,
	Dtype* code_diff, Dtype* loss_diff){
	for (int pos = 0; pos < batch_*dim_; ++pos){
		if (code[pos] > 0){
			if (res_code[pos] < 1){
				code_diff[pos] = top_diff[pos];
				loss_diff[pos] = top_diff[pos] / lamda_;
			}
			else{
				code_diff[pos] = Dtype(0.0);
				loss_diff[pos] = Dtype(0.0);
			}
		}
		else{
			if (res_code[pos] > -1){
				code_diff[pos] = top_diff[pos];
				loss_diff[pos] = -top_diff[pos] / lamda_;
			}
			else{
				code_diff[pos] = Dtype(0.0);
				loss_diff[pos] = Dtype(0.0);
			}
		}
	}
}

template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_code = bottom[0]->cpu_data(); // hash code
	const Dtype* bottom_loss = bottom[1]->cpu_data(); // quantization loss
	Dtype* top_data = top[0]->mutable_cpu_data(); // add the qunatiztion loss hash code
	vector<int> top_shape = top[0]->shape();
	if (sigmoid_flag_)
		quantization_add_forward_cpu_sig(bottom_code, bottom_loss, top_data);
	else
		quantization_add_forward_cpu_tanh(bottom_code, bottom_loss, top_data);

}

template <typename Dtype>
void ResidualQuantizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom){
	const Dtype* bottom_code = bottom[0]->cpu_data();
	const Dtype* bottom_loss = bottom[1]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	Dtype* bottom_code_diff = bottom[0]->mutable_cpu_diff();
	Dtype* bottom_loss_diff = bottom[1]->mutable_cpu_diff();
	if (sigmoid_flag_)
		quantization_add_backward_cpu_sig(bottom_code, bottom_loss,
		top_diff, top_data,
		bottom_code_diff, bottom_loss_diff);
	else
		quantization_add_backward_cpu_tanh(bottom_code, bottom_loss,
		top_diff, top_data,
		bottom_code_diff, bottom_loss_diff);
}




INSTANTIATE_CLASS(ResidualQuantizationLayer);
REGISTER_LAYER_CLASS(ResidualQuantization);
}// namespace caffe