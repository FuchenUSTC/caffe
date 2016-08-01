#include <vector>
#include <cstring>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/video_temporal_pooling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// added by Fuchen Long for the 
// video temproal pooling layer test 
// added in 8/1/2016

namespace caffe{

template <typename TypeParam>
class VideoTemporalPoolingLayerTest : public MultiDeviceTest<TypeParam>{
	typedef typename TypeParam::Dtype Dtype;
protected:
	VideoTemporalPoolingLayerTest()
		:blob_bottom_(new Blob<Dtype>()),
		 blob_top_(new Blob<Dtype>()){
		// init the video pooling layer
		vector<int> bottom_shape = vector<int>(2, 0);
		bottom_shape[0] = 6;
		bottom_shape[1] = 3;
		blob_bottom_->Reshape(bottom_shape);
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_);
		blob_bottom_vec_.push_back(blob_bottom_);
		blob_top_vec_.push_back(blob_top_);
	}

	virtual ~VideoTemporalPoolingLayerTest(){
		delete blob_bottom_; 
		delete blob_top_;
	}

	void TestForward(){
		// Set the frame number
		LayerParameter layer_param;
		VideoTemporalPoolingParameter* pool_param = layer_param.mutable_video_temporal_pooling_param();
		pool_param->set_frame_num(3);
		VideoTemporalPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		EXPECT_EQ(this->blob_top_->shape().size(), 4);
		EXPECT_EQ(this->blob_top_->num(), 2);
		EXPECT_EQ(this->blob_top_->channels(), 3);
		EXPECT_EQ(this->blob_top_->height(), 1);
		EXPECT_EQ(this->blob_top_->width(), 3);
	}

	Blob<Dtype>*  blob_bottom_;
	Blob<Dtype>*  blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(VideoTemporalPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(VideoTemporalPoolingLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(VideoTemporalPoolingLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	VideoTemporalPoolingParameter* pool_param = layer_param.mutable_video_temporal_pooling_param();
	pool_param->set_frame_num(3);
	VideoTemporalPoolingLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-2);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}

}