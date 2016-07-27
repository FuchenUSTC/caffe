#ifndef CAFFE_PAIRWISE_CLIP_HINGE_LOSS_LAYER_HPP_
#define CAFFE_PAIRWISE_CLIP_HINGE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
	
/**
* @ added by Fuchen Long, used in hashing learning 
*
* TODO(dox): thorough documentation for Forward, Backward, and proto params.
* 
* In 7.26.2016
*/

template<typename Dtype>
class PairWiseClipHingeLossLayer : public LossLayer<Dtype>{
public:
	explicit PairWiseClipHingeLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param), diff_() {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	virtual inline const char* type() const { return "PairWiseClipHingeLoss"; }
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return bottom_index != 3;
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_dowm, const vector<Blob<Dtype>*>& bottom);

	void average_hashing(const vector<Blob<Dtype>*>& bottom);
	Dtype compute_pairwiseloss(int batchsize, int Dimv, const vector<Blob<Dtype>*>& bottom);
	Dtype compute_structureloss(const vector<Blob<Dtype>*>& bottom);
	void compute_gradient_structure(int index, int hash_pos);

	int dim;
	int frame_num;
	int batch;
	Dtype margin;
	Dtype lamda;
	Blob<Dtype> diff_;
	Blob<Dtype> dist_sq_;
	Blob<Dtype> diff_sub_or_di; // F-F-
	Blob<Dtype> diff_pow_or_di; // ||F-F-||2
	Blob<Dtype> ave_or, ave_di; // Aver(F,F-)
	Blob<Dtype> sub_or, sub_di; // Subcessive(F,F-)
	Blob<Dtype> pow_sub_or, pow_sub_di; // PowX(sub(F,F-))
	Blob<Dtype> gradient_pairwise;
	Blob<Dtype> gradient_structure;
	Blob<Dtype> gradient;
};


} // namespace caffe

#endif