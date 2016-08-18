#include <algorithm>
#include <vector>

#include "caffe/layers/pairwise_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long for hashing learning

namespace caffe{

template<typename Dtype>
void PairWiseCrossEntropyLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	lamda_ = this->layer_param_.pairwise_cross_entropy_loss_param().lamda();
	dim_ = bottom[0]->channels();
	batch_ = bottom[0]->num();

	CHECK_EQ(bottom[1]->num(), batch_)
		<< "PAIRWISE_CROSS_ENTROPY_LOSS: the batch size should match.";
	CHECK_EQ(bottom[2]->num(), batch_)
		<< "PAIRWISE_CROSS_ENTROPY_LOSS: the label batch should match.";
	CHECK_EQ(bottom[1]->channels(), dim_)
		<< "PAIRWISE_CROSS_ENTROPY_LOSS: the dim_ of each other should match.";

	inner_product_.Reshape(batch_, 1, 1, 1);
	inner_exp_.Reshape(batch_, 1, 1, 1);
	meta_data_.Reshape(1, dim_, 1, 1);
	gradient_1_.Reshape(1, dim_, 1, 1);
	gradient_2_.Reshape(1, dim_, 1, 1);
	meta_g_1_.Reshape(1, dim_, 1, 1);
	meta_g_2_.Reshape(1, dim_, 1, 1);

}

template<typename Dtype>
void PairWiseCrossEntropyLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data_1 = bottom[0]->cpu_data();
	const Dtype* bottom_data_2 = bottom[1]->cpu_data();
	const Dtype* label_data = bottom[2]->cpu_data();
	Dtype* inner_mutable_data = inner_product_.mutable_cpu_data();
	caffe_set(batch_, Dtype(0.0), inner_mutable_data);

	Dtype pairwise_loss(0.0);
	
 	for (int n = 0; n < batch_; ++n){
		int offset = n*dim_;
		caffe_cpu_gemv(CblasNoTrans, 1, dim_, Dtype(1.0), bottom_data_1 + offset,
			bottom_data_2 + offset, Dtype(0.0), inner_mutable_data + n);
		Dtype inner = inner_product_.cpu_data()[n];
		pairwise_loss += (log(1 + inner) - label_data[n] * inner);
	}

	top[0]->mutable_cpu_data()[0] = pairwise_loss / batch_;
	
}


template<typename Dtype>
void PairWiseCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	Dtype* bottom_diff_1 = bottom[0]->mutable_cpu_diff();
	Dtype* bottom_diff_2 = bottom[1]->mutable_cpu_diff();
	const Dtype* bottom_data_1 = bottom[0]->cpu_data();
	const Dtype* bottom_data_2 = bottom[1]->cpu_data();
	const Dtype* inner_product_data = inner_product_.cpu_data();
	const Dtype* label_data = bottom[2]->cpu_data();
	
	if (propagate_down[0]){
		for (int n = 0; n < batch_; ++n){
			int offset = n*dim_;
			Dtype inner = inner_product_data[n];
			caffe_copy(dim_, bottom_data_2 + offset, gradient_1_.mutable_cpu_data()); // inverse
			caffe_copy(dim_, bottom_data_1 + offset, gradient_2_.mutable_cpu_data());

			caffe_copy(dim_, bottom_data_2 + offset, meta_g_1_.mutable_cpu_data()); // inverse
			caffe_copy(dim_, bottom_data_1 + offset, meta_g_2_.mutable_cpu_data());

			caffe_scal(dim_, inner / (Dtype(1.0) + inner), gradient_1_.mutable_cpu_data());
			caffe_scal(dim_, inner / (Dtype(1.0) + inner), gradient_2_.mutable_cpu_data());

			caffe_scal(dim_, label_data[n], meta_g_1_.mutable_cpu_data());
			caffe_scal(dim_, label_data[n], meta_g_2_.mutable_cpu_data());

			caffe_sub(dim_, gradient_1_.cpu_data(), meta_g_1_.cpu_data(), bottom_diff_1 + offset);
			caffe_sub(dim_, gradient_2_.cpu_data(), meta_g_2_.cpu_data(), bottom_diff_2 + offset);
		}
		caffe_scal(dim_*batch_, Dtype(2.0), bottom_diff_1);
		caffe_scal(dim_*batch_, Dtype(2.0), bottom_diff_2);
	}
}

	INSTANTIATE_CLASS(PairWiseCrossEntropyLossLayer);
	REGISTER_LAYER_CLASS(PairWiseCrossEntropyLoss);
}// namespace caffe
