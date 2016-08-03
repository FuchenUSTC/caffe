#include <algorithm>
#include <vector>

#include "caffe/layers/orthogonal_constraint_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

/* Briefly for the deep hashing learning
* Added by Fuchen Long in 8/3/2016
*/
namespace caffe{

template <typename Dtype>
void OrthogonalConstraintLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	dim_ = this->layer_param_.orthogonal_constraint_loss_loss_param().dim();
	alpha_ = this->layer_param_.orthogonal_constraint_loss_loss_param().alpha();
	batch_ = bottom[0]->num();
	CHECK_EQ(dim_, bottom[0]->channels())
		<< "OrthogonalConstraintLoss must have the same dim.";

	unit_matrix_.Reshape(dim_, dim_, 1, 1);
	code_multi_matrix_.Reshape(dim_, dim_, 1, 1);
	code_t_matrix_.Reshape(dim_, dim_, 1, 1);
	gradient_.Reshape(dim_, batch_, 1, 1);

	Dtype* unit_data = unit_matrix_.mutable_cpu_data();
	for (int i = 0; i < dim_; ++i){
		for (int j = 0; j < dim_; ++j){
			if (j == i) unit_data[j + i*dim_] = Dtype(dim_);
			else unit_data[j + i*dim_] = Dtype(0.0);
		}
	}
	caffe_set(dim_*dim_, Dtype(0.0), code_multi_matrix_.mutable_cpu_data());
	caffe_set(dim_*dim_, Dtype(0.0), code_t_matrix_.mutable_cpu_data());
	caffe_set(dim_*batch_, Dtype(0.0), gradient_.mutable_cpu_data());
}

template <typename Dtype>
void OrthogonalConstraintLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	CHECK_EQ(bottom[0]->channels(), dim_)
		<< "OrthogonalConstraintLoss: code length must match.";
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void OrthogonalConstraintLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){

	Dtype orthogonalloss(0.0);
	const Dtype* unit_data = unit_matrix_.cpu_data();
	Dtype* code_multi_mutable_data = code_multi_matrix_.mutable_cpu_data();
	const Dtype* code_multi_data = code_multi_matrix_.cpu_data();
	Dtype* code_t_mutable_data = code_t_matrix_.mutable_cpu_data();
	const Dtype* code_t_data = code_t_matrix_.cpu_data();

	caffe_cpu_gemm(CblasTrans, CblasNoTrans, dim_, dim_, batch_, Dtype(1.0),
		bottom[0]->cpu_data(), bottom[0]->cpu_data(), Dtype(0.0), code_multi_mutable_data); // X'X
	caffe_sub(dim_*dim_, code_multi_data, unit_data, code_multi_mutable_data); // X'X - I
	caffe_powx(dim_*dim_, code_multi_data, Dtype(2.0), code_t_mutable_data); // ||X'X - I||^2
	orthogonalloss = caffe_cpu_asum(dim_*dim_, code_t_data) / (alpha_*batch_);
	top[0]->mutable_cpu_data()[0] = orthogonalloss;
}

template <typename Dtype>
void OrthogonalConstraintLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

	const Dtype* unit_data = unit_matrix_.cpu_data();
	Dtype* code_multi_mutable_data = code_multi_matrix_.mutable_cpu_data();
	const Dtype* code_multi_data = code_multi_matrix_.cpu_data();
	Dtype* code_t_mutable_data = code_t_matrix_.mutable_cpu_data();
	const Dtype* code_t_data = code_t_matrix_.cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* gradient_mutable_data = gradient_.mutable_cpu_data();
	const Dtype* gradient_data = gradient_.cpu_data();

	caffe_set(dim_*dim_, Dtype(0.0), code_multi_matrix_.mutable_cpu_data());

	if (propagate_down[0]){
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, dim_, dim_, batch_, Dtype(1.0),
			bottom[0]->cpu_data(), bottom[0]->cpu_data(), Dtype(0.0), code_multi_mutable_data); // X'X
		caffe_sub(dim_*dim_, code_multi_data, unit_data, code_multi_mutable_data); // X'X - I
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, dim_, batch_, dim_, Dtype(1.0),
			code_multi_data, bottom[0]->cpu_data(), Dtype(0.0), gradient_mutable_data); // (X'X - I)X'
		caffe_scal(dim_*batch_, Dtype(8) / (batch_*alpha_), gradient_mutable_data); // scale the size 
		int res_offset = 0;
		for (int c = 0; c < dim_; ++c){
			for (int n = 0; n < batch_; ++n){
				bottom_diff[n*dim_ + c] = gradient_data[res_offset];
				res_offset += 1;
			}
		}
	}
}

INSTANTIATE_CLASS(OrthogonalConstraintLossLayer);
REGISTER_LAYER_CLASS(OrthogonalConstraintLoss);

}// namespace caffe