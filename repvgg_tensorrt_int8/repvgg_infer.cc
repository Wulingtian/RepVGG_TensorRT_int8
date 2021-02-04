
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include  "my_interface.h"
#include <chrono>
#include <vector>
#include <npp.h>

#define INPUT_W 224
#define INPUT_H 224
#define NUM_CLASS 2
char* output_name = "100";
char* trt_model_path = "../models/RepVGG-A0_int8.trt";
std::string test_img = "../test_imgs/dog.33.jpg";
std::vector<float> img_mean = {0.485, 0.456, 0.406};
std::vector<float> img_std = { 0.229, 0.224, 0.225 };

using namespace cv;
using namespace std;

void cudaResize(cv::Mat &image, cv::Mat &rsz_img)
{
    int outsize = rsz_img.cols * rsz_img.rows * sizeof(uchar3);

    int inwidth = image.cols;
    int inheight = image.rows;
    int memSize = inwidth * inheight * sizeof(uchar3);

    NppiSize srcsize = {inwidth, inheight};
    NppiRect srcroi  = {0, 0, inwidth, inheight};
    NppiSize dstsize = {rsz_img.cols, rsz_img.rows};
    NppiRect dstroi  = {0, 0, rsz_img.cols, rsz_img.rows};

    uchar3* d_src = NULL;
    uchar3* d_dst = NULL;
    cudaMalloc((void**)&d_src, memSize);
    cudaMalloc((void**)&d_dst, outsize);
    cudaMemcpy(d_src, image.data, memSize, cudaMemcpyHostToDevice);

    // nvidia npp 图像处理
    nppiResize_8u_C3R( (Npp8u*)d_src, inwidth * 3, srcsize, srcroi,
                       (Npp8u*)d_dst, rsz_img.cols * 3, dstsize, dstroi,
                       NPPI_INTER_LINEAR );


    cudaMemcpy(rsz_img.data, d_dst, outsize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

std::vector<float> prepareImage(cv::Mat &src_img) {
    std::vector<float> result(INPUT_W * INPUT_H * 3);
    float *data = result.data();
    float ratio = float(INPUT_W) / float(src_img.cols) < float(INPUT_H) / float(src_img.rows) ? float(INPUT_W) / float(src_img.cols) : float(INPUT_H) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    cv::Mat rsz_img = cv::Mat::zeros(cv::Size(src_img.cols*ratio, src_img.rows*ratio), CV_8UC3);
    cudaResize(src_img, rsz_img);
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    int channelLength = INPUT_W * INPUT_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data)
    };


    auto pr_start = std::chrono::high_resolution_clock::now();
    cv::split(flt_img, split_img);
    for (int i = 0; i < 3; i++) {
            split_img[i] = (split_img[i]/255 - img_mean[i]) / img_std[i];
    }
    auto pr_end = std::chrono::high_resolution_clock::now();

    auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;
    return result;
}


int main(int argc, const char *argv[]) {
    float total = 0, ms, pr_ms, po_ms;
    int test_echo = 20;

    // 创建输入输出tensor结构体
    tensor_params_array_t in_tensor_params_ar = {0};
    tensor_params_array_t out_tensor_params_ar = {0};
    tensor_array_t *input_tensor_array = NULL;
    tensor_array_t *ouput_tensor_array = NULL;

    /****************** */
    // 定义输入tensor
    in_tensor_params_ar.nArraySize = 1;
    in_tensor_params_ar.pTensorParamArray = (tensor_params_t *) malloc(
            in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    memset(in_tensor_params_ar.pTensorParamArray, 0, in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    tensor_params_t *cur_in_tensor_params = in_tensor_params_ar.pTensorParamArray;

    // 第一个输入tensor
    cur_in_tensor_params[0].nDims = 4;
    cur_in_tensor_params[0].type = DT_FLOAT;
    cur_in_tensor_params[0].pShape[0] = 1; //batch size can't set to -1
    cur_in_tensor_params[0].pShape[1] = 3;
    cur_in_tensor_params[0].pShape[2] = INPUT_H;
    cur_in_tensor_params[0].pShape[3] = INPUT_W;
    strcpy(cur_in_tensor_params[0].aTensorName, "input.1");
    cur_in_tensor_params[0].tensorMemoryType = CPU_MEM_ALLOC;

    /*************** */
    // 定义输出tensor
    out_tensor_params_ar.nArraySize = 1;
    out_tensor_params_ar.pTensorParamArray = (tensor_params_t *) malloc(
            out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    memset(out_tensor_params_ar.pTensorParamArray, 0, out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    tensor_params_t *cur_out_tensor_params = out_tensor_params_ar.pTensorParamArray;

    cur_out_tensor_params[0].nDims = 2;
    cur_out_tensor_params[0].type = DT_FLOAT;
    cur_out_tensor_params[0].pShape[0] = 1;
    cur_out_tensor_params[0].pShape[1] = NUM_CLASS;
    cur_out_tensor_params[0].tensorMemoryType = CPU_MEM_ALLOC;
    strcpy(cur_out_tensor_params[0].aTensorName, output_name);

    // 初始化输入输出结构体，分配内存
    if (my_init_tensors(&in_tensor_params_ar, &out_tensor_params_ar,
                        &input_tensor_array, &ouput_tensor_array) != MY_SUCCESS) {
        printf("Open Internal memory error!\n");
    }

    //===================obtain Handle=========================================
    model_params_t tModelParam = {0}; //model input parameter
    model_handle_t tModelHandle = {0};

    strcpy(tModelParam.visibleCard, "0");
    tModelParam.gpu_id = 0; //GPU 0
    tModelParam.bIsCipher = FALSE;
    tModelParam.maxBatchSize = 1;

    strcpy(tModelParam.model_path, trt_model_path);

    //call API open model
    if (my_load_model(&tModelParam,
                      input_tensor_array,
                      ouput_tensor_array,
                      &tModelHandle) != MY_SUCCESS) {
        printf("Open model error!\n");
    }
    std::cout << "Load model sucess\n";


    string file_name = test_img;
    tensor_t *cur_input_tensor_image = &(input_tensor_array->pTensorArray[0]);

    cv::Mat cImage;
    cImage = cv::imread(file_name);
    std::cout << "Read img finished!\n";
    cv::Mat showImage = cImage.clone();


    auto pr_start = std::chrono::high_resolution_clock::now();
    vector<float> pr_img = prepareImage(cImage);
    auto pr_end = std::chrono::high_resolution_clock::now();
    pr_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();

    memcpy((float *) (cur_input_tensor_image->pValue),
           pr_img.data(), 3 * INPUT_H * INPUT_W * sizeof(float));


    printf("----->memcpy data is success......\n");
    for (int j = 0; j < test_echo; ++j) {
        auto t_start = std::chrono::high_resolution_clock::now();

        my_inference_tensors(&tModelHandle);

        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;
        std::cout << "[ " << j << " ] " << ms << " ms." << std::endl;
    }

    total /= test_echo;
    std::cout << "Average over " << test_echo << " runs is " << total << " ms." << std::endl;

    tensor_t *cur_output_tensor = &(ouput_tensor_array->pTensorArray[0]);
    float * output = static_cast<float *>(cur_output_tensor->pValue);

    int outSize = cur_output_tensor->pTensorInfo->nElementSize;
    //std::cout << "outSize:" << outSize << std::endl;

    int index = 0;
    float max = output[0];
    for (int i = 0; i < outSize; i++) {
        if (max < output[i]) {
            max = output[i];
            index = i; 
        }
    }       
    
    std::cout << "prob: " << index << std::endl;
    //std::cout << "prob:" << output[0] << " " << output[1] << std::endl;

    my_deinit_tensors(input_tensor_array, ouput_tensor_array);

    my_release_model(&tModelHandle);



    std::cout << "complete!!!" << std::endl;

    return 0;
}
