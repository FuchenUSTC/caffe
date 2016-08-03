#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/orthogonal_constraint_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// Added by Fuchen Long in 8/3/2016
// to test orthogonal_constraint_loss_layer

namespace caffe{

template <typename TypeParam>
class OrthogonalConstraintLossLayerTest :public MultiDeviceTest<TypeParam>{
protected:
	OrthogonalConstraintLossLayerTest()
		:blob_bottom_data_(new Blob<Dtype>(12, 24, 1, 1)),
		blob_top_loss_(new Blob<Dtype>()){

		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_data_);
		blob_bottom_vec_.push_back(blob_bottom_data_);
		blob_top_vec_.push_back(blob_top_loss_);
	}

	virtual ~OrthogonalConstraintLossLayerTest(){
		delete blob_bottom_data_;
		delete blob_top_loss_;
	}

	void TestForward(){
		LayerParameter layer_param;
		OrthogonalConstraintLossParameter* ortho_param = 
			layer_param.mutable_orthogonal_constraint_loss_loss_param();
		ortho_param->set_dim(24);
		ortho_param->set_alpha(100);
		LayerParameter layer_param2;
		OrthogonalConstraintLossParameter* ortho_param2 =
			layer_param2.mutable_orthogonal_constraint_loss_loss_param();
		ortho_param2->set_dim(24);
		ortho_param2->set_alpha(100);

		layer_param.add_loss_weight(1.0);
		
		OrthogonalConstraintLossLayer<Dtype> layer_weight_1(layer_param);
		layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		Dtype loss_weight_1 =
			layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		const Dtype kLossWeight = 7.7;
		layer_param2.add_loss_weight(kLossWeight);
		OrthogonalConstraintLossLayer<Dtype> layer_weight_2(layer_param2);
		layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_2 =
			layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype kErrorMargin = 1e-5;
		EXPECT_NEAR(loss_weight_1*kLossWeight, loss_weight_2, kErrorMargin);
		//Make sure the loss is non-trivial
		const Dtype kNonTrivialAbsThresh = 1e-1;
		EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
	}

	Blob<Dtype>* const blob_bottom_data_;
	Blob<Dtype>* const blob_top_loss_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OrthogonalConstraintLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(OrthogonalConstraintLossLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(OrthogonalConstraintLossLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	const Dtype kLossWeight = 3.7;
	layer_param.add_loss_weight(kLossWeight);
	OrthogonalConstraintLossParameter* ortho_param =
		layer_param.mutable_orthogonal_constraint_loss_loss_param();
	ortho_param->set_dim(24);
	ortho_param->set_alpha(100);

	OrthogonalConstraintLossLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}
} // namespace caffe