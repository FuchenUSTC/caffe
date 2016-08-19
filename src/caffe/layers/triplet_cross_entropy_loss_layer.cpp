#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long for hashing in 8/19/2016

namespace caffe{

template <typename Dtype>
void TripletCrossEntropyLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	dim_ = this->layer_param_.triplet_cross_entropy_loss_param().dim();
	batch_ = bottom[0]->num();

	CHECK_EQ(bottom[1]->num(), batch_);
	CHECK_EQ(bottom[2]->num(), batch_);

	CHECK_EQ(bottom[0]->channels(), dim_);
	CHECK_EQ(bottom[1]->channels(), dim_);
	CHECK_EQ(bottom[2]->channels(), dim_)
		<< "TRIPLET_CROSS_ENTROPY_LOSS: the dim_ of each ohter should match.";

	inner_product_q_s_.Reshape(batch_, 1, 1, 1);
	inner_product_q_d_.Reshape(batch_, 1, 1, 1);
	exp_qmeta_q_s_.Reshape(1, dim_, 1, 1);
	exp_qmeta_q_d_.Reshape(1, dim_, 1, 1);
	gradient_q_.Reshape(1, dim_, 1, 1);
	exp_smeta_q_s_.Reshape(1, dim_, 1, 1);
	gradient_s_.Reshape(1, dim_, 1, 1);
	gradient_d_.Reshape(1, dim_, 1, 1);

}

template <typename Dtype>
void TripletCrossEntropyLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data_1 = bottom[0]->cpu_data();
	const Dtype* bottom_data_2 = bottom[1]->cpu_data();
	const Dtype* bottom_data_3 = bottom[2]->cpu_data();
	Dtype* inner_mutable_q_s = inner_product_q_s_.mutable_cpu_data();
	Dtype* inner_mutable_q_d = inner_product_q_d_.mutable_cpu_data();
	caffe_set(batch_, Dtype(0.0), inner_mutable_q_s);
	caffe_set(batch_, Dtype(0.0), inner_mutable_q_d);

	Dtype triplet_loss(0.0);

	for (int n = 0; n < batch_; ++n){
		int offset = n*dim_;
		caffe_cpu_gemv(CblasNoTrans, 1, dim_, Dtype(1.0), bottom_data_1 + offset,
			bottom_data_2 + offset, Dtype(0.0), inner_mutable_q_s + n);
		caffe_cpu_gemv(CblasNoTrans, 1, dim_, Dtype(1.0), bottom_data_1 + offset,
			bottom_data_3 + offset, Dtype(0.0), inner_mutable_q_d + n);
		Dtype inner_q_s = inner_product_q_s_.cpu_data()[n];
		Dtype inner_q_d = inner_product_q_d_.cpu_data()[n];
		triplet_loss += (log(1 + exp(inner_q_s)) +
			log(1 + exp(inner_q_d)) - inner_q_s);
	}
	top[0]->mutable_cpu_data()[0] = triplet_loss / batch_;
}

template <typename Dtype>
void TripletCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	Dtype* bottom_diff_1 = bottom[0]->mutable_cpu_diff();
	Dtype* bottom_diff_2 = bottom[1]->mutable_cpu_diff();
	Dtype* bottom_diff_3 = bottom[2]->mutable_cpu_diff();
	const Dtype* bottom_data_1 = bottom[0]->cpu_data();
	const Dtype* bottom_data_2 = bottom[1]->cpu_data();
	const Dtype* bottom_data_3 = bottom[2]->cpu_data();
	const Dtype* inner_data_q_s = inner_product_q_s_.cpu_data();
	const Dtype* inner_data_q_d = inner_product_q_d_.cpu_data();

	if (propagate_down[0]){
		for (int n = 0; n < batch_; ++n){
			int offset = n*dim_;
			Dtype inner_q_s = inner_data_q_s[n];
			Dtype inner_q_d = inner_data_q_d[n];

			// gradient for query sample
			caffe_copy(dim_, bottom_data_2 + offset, exp_qmeta_q_s_.mutable_cpu_data());
			caffe_copy(dim_, bottom_data_3 + offset, exp_qmeta_q_d_.mutable_cpu_data());
			caffe_scal(dim_, exp(inner_q_s) / (Dtype(1.0) + exp(inner_q_s)), exp_qmeta_q_s_.mutable_cpu_data());
			caffe_scal(dim_, exp(inner_q_d) / (Dtype(1.0) + exp(inner_q_d)), exp_qmeta_q_d_.mutable_cpu_data());
			caffe_add(dim_, exp_qmeta_q_s_.cpu_data(), exp_qmeta_q_d_.cpu_data(), gradient_q_.mutable_cpu_data());
			caffe_sub(dim_, gradient_q_.cpu_data(), bottom_data_2 + offset, gradient_q_.mutable_cpu_data());

			// gradient for positive sample
			caffe_copy(dim_, bottom_data_1 + offset, exp_smeta_q_s_.mutable_cpu_data());
			caffe_scal(dim_, exp(inner_q_s) / (Dtype(1.0) + exp(inner_q_s)), exp_smeta_q_s_.mutable_cpu_data());
			caffe_sub(dim_, exp_smeta_q_s_.cpu_data(), bottom_data_1 + offset, gradient_s_.mutable_cpu_data());

			// gradient for negative sample
			caffe_copy(dim_, bottom_data_1 + offset, gradient_d_.mutable_cpu_data());
			caffe_scal(dim_, exp(inner_q_d) / (Dtype(1.0) + exp(inner_q_d)), gradient_d_.mutable_cpu_data());

			// put to the bottom diff
			caffe_copy(dim_, gradient_q_.cpu_data(), bottom_diff_1 + offset);
			caffe_copy(dim_, gradient_s_.cpu_data(), bottom_diff_2 + offset);
			caffe_copy(dim_, gradient_d_.cpu_data(), bottom_diff_3 + offset);
		}
		caffe_scal(dim_*batch_, Dtype(2.0) / batch_, bottom_diff_1);
		caffe_scal(dim_*batch_, Dtype(2.0) / batch_, bottom_diff_2);
		caffe_scal(dim_*batch_, Dtype(2.0) / batch_, bottom_diff_3);
	}

}
	INSTANTIATE_CLASS(TripletCrossEntropyLossLayer);
	REGISTER_LAYER_CLASS(TripletCrossEntropyLoss);
}// namspace caffe