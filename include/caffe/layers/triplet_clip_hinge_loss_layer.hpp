#ifndef CAFFE_TRIPLET_CLIP_HINGE_LOSS_LAYER
#define CAFFE_TRIPLET_CLIP_HINGE_LOSS_LAYER

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
* In 7.28.2016
*/

template<typename Dtype>
class TripletClipHingeLossLayer : public LossLayer<Dtype>
{
public:
	explicit TripletClipHingeLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param), diff_() {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	virtual inline const char* type() const{ return "TripletClipHingeLoss"; }
	virtual inline bool AllowForceBackward(const int bottom_index) const
	{
		return bottom_index != 3;

	}
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_dowm, const vector<Blob<Dtype>*>& bottom);
	void average_hashing(const vector<Blob<Dtype>*>& bottom);
	Dtype compute_tripletloss(int batchsize, int Dimv);
	Dtype compute_structureloss(const vector<Blob<Dtype>*>& bottom);
	void compute_gradient_structure(int index, int hash_pos);
	int dim;
	int frame_num;
	int batch;
	Dtype margin;
	Dtype lamda;
	Blob<Dtype> diff_;
	Blob<Dtype> dist_sq_;
	Blob<Dtype> diff_sub_or_si; // F-F+
	Blob<Dtype> diff_sub_or_di; // F-F-
	Blob<Dtype> diff_pow_or_si; // ||F-F+||2
	Blob<Dtype> diff_pow_or_di; // ||F-F-||2
	Blob<Dtype> ave_or, ave_si, ave_di; // Aver(F,F+,F-)
	Blob<Dtype> sub_or, sub_si, sub_di; // Subcessive(F,F+,F-)
	Blob<Dtype> pow_sub_or, pow_sub_si, pow_sub_di; // PowX(sub(F,F+,F-))
	Blob<Dtype> gradient_triplet;
	Blob<Dtype> gradient_structure;
	Blob<Dtype> gradient;
};

}
#endif