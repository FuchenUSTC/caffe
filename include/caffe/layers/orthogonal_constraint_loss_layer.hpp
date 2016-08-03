#ifndef CAFFE_ORTHOGONAL_CONSTRAINT_LOSS_LAYER_HPP_
#define CAFFE_ORTHOGONAL_CONSTRAINT_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
	/*
	** @ briefly to make the hash bits more balance
	**   for orthogonal cosntraint.
	**   Added by Fuchen Long in 8/3/2016
	*/

template <typename Dtype>
class OrthogonalConstraintLossLayer :public LossLayer<Dtype>{
public:
	explicit OrthogonalConstraintLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "OrthogonalConstraintLoss"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	// copydoc OrthogonalConstraintLossLayer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int dim_;
	int batch_;
	Dtype alpha_;

	Blob<Dtype> unit_matrix_;
	Blob<Dtype> code_multi_matrix_;
	Blob<Dtype> code_t_matrix_;
	Blob<Dtype> gradient_;
};

}

#endif